/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kv_cache_manager_v2/storage/core.h"

#include "tensorrt_llm/common/assert.h"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unistd.h>
#include <unordered_map>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// SlotAllocator
// ---------------------------------------------------------------------------

SlotAllocator::SlotAllocator(SlotCount capacity)
    : mCapacity(capacity)
    , mTargetCapacity(capacity)
    , mNumActiveSlots(0)
    , mNumReadyRecycledSlots(0)
    , mOccupiedMask(slotCountToSizeT(capacity))
{
}

SlotAllocator::~SlotAllocator()
{
    // Mirrors Python SlotAllocator.__del__ (assert_critical checks).
    if (TLLM_UNLIKELY(gDebug))
    {
        KVCM2_CHECK_FATAL_WITH_INFO(slotCountToSizeT(mNumReadyRecycledSlots) == mRecycledSlots.size(),
            "SlotAllocator destroyed with unfinished events — did you call synchronize()?");
        KVCM2_CHECK_FATAL_WITH_INFO(mTargetCapacity == mCapacity && mOverflowSlots.empty(),
            "SlotAllocator destroyed while resize is in progress");
        KVCM2_CHECK_FATAL_WITH_INFO(
            mOccupiedMask.numSetBits() == 0, "SlotAllocator destroyed with occupied slots still in use");
        KVCM2_CHECK_FATAL_WITH_INFO(mRecycledSlots.size() == slotCountToSizeT(mNumActiveSlots),
            "SlotAllocator destroyed with some slots not recycled");
    }

    // Recycled and overflow slots still carry their ids; the allocator owns them, so clear them
    // here rather than letting ~Slot report each one. A slot that escaped the allocator is not in
    // the recycle lists and is still reported.
    for (auto& slot : mRecycledSlots)
    {
        slot.resetSlot();
    }
    for (auto& slot : mOverflowSlots)
    {
        slot.resetSlot();
    }
}

SlotCount SlotAllocator::numFreeSlots() const noexcept
{
    SlotCount const inactiveSlots = mTargetCapacity > mNumActiveSlots ? mTargetCapacity - mNumActiveSlots : 0;
    return slotCountValueFromSize(mRecycledSlots.size()) + inactiveSlots;
}

SlotCount SlotAllocator::numOccupiedSlots() const noexcept
{
    return slotCountValueFromSize(mOccupiedMask.numSetBits());
}

Slot SlotAllocator::allocate()
{
    if (numFreeSlots() == 0)
    {
        throw OutOfPagesError("SlotAllocator: no free slots");
    }
    scrubEvents();

    Slot slot;
    if (mNumReadyRecycledSlots > 0)
    {
        TLLM_CHECK_DEBUG_WITH_INFO(!mRecycledSlots.empty(), "ready recycled slots > 0 but deque is empty");
        slot = std::move(mRecycledSlots.front());
        mRecycledSlots.pop_front();
        TLLM_CHECK_DEBUG_WITH_INFO(slot.hasValidSlot(), "ready recycled slot has no valid id");
        --mNumReadyRecycledSlots;
        TLLM_CHECK_DEBUG_WITH_INFO(slot.readyEvent.isClosed(), "ready recycled slot has non-null event");
    }
    else if (mNumActiveSlots < std::min(mCapacity, mTargetCapacity))
    {
        slot.setSlotId(SlotId{mNumActiveSlots++});
    }
    else
    {
        slot = std::move(mRecycledSlots.front());
        mRecycledSlots.pop_front();
        TLLM_CHECK_DEBUG_WITH_INFO(slot.hasValidSlot(), "non-ready recycled slot has no valid id");
    }
    mOccupiedMask.set(toSizeT(slot.slotId()));
    return slot;
}

std::vector<Slot> SlotAllocator::allocateMultiple(SlotCount numSlots)
{
    if (numSlots < 0)
    {
        throw LogicError("SlotAllocator::allocateMultiple: slot count must be non-negative");
    }
    if (numFreeSlots() < numSlots)
    {
        throw OutOfPagesError("SlotAllocator: not enough free slots");
    }
    std::vector<Slot> result;
    result.reserve(slotCountToSizeT(numSlots));
    for (SlotCount slotCount{0}; slotCount < numSlots; ++slotCount)
    {
        result.push_back(allocate());
    }
    return result;
}

void SlotAllocator::release(Slot slot)
{
    if (!slot.hasValidSlot())
    {
        throw LogicError("SlotAllocator::release: slot has no valid id");
    }
    SlotId const slotId = slot.slotId();
    if (!isValidSlotId(slotId) || !mOccupiedMask.get(toSizeT(slotId)))
    {
        // The id is not one this allocator has outstanding, so drop it rather than let the
        // rejected slot look like an unreleased one.
        slot.resetSlot();
        throw LogicError("SlotAllocator::release: slot is not occupied");
    }
    mOccupiedMask.clear(toSizeT(slotId));
    if (slotId < mTargetCapacity)
    {
        mRecycledSlots.push_back(std::move(slot));
    }
    else
    {
        mOverflowSlots.push_back(std::move(slot));
    }
    scrubEvents();
    TLLM_CHECK_DEBUG(check());
}

void SlotAllocator::expand(SlotCount newNumSlots)
{
    TLLM_CHECK_DEBUG(check());
    TLLM_CHECK_DEBUG(mTargetCapacity == mCapacity);
    TLLM_CHECK_DEBUG(newNumSlots > mCapacity);
    mOccupiedMask.resize(slotCountToSizeT(newNumSlots));
    mCapacity = newNumSlots;
    mTargetCapacity = newNumSlots;
    TLLM_CHECK_DEBUG(check());
}

void SlotAllocator::prepareForShrink(SlotCount newNumSlots)
{
    TLLM_CHECK_DEBUG(check());
    TLLM_CHECK_DEBUG(mTargetCapacity == mCapacity);
    TLLM_CHECK_DEBUG(newNumSlots < mCapacity);
    std::deque<Slot> newRecycled;
    SlotCount newNumReady = 0;
    SlotCount const oldNumReady = mNumReadyRecycledSlots;
    SlotCount idx = 0;
    for (auto& s : mRecycledSlots)
    {
        if (s.slotId() < newNumSlots)
        {
            if (idx < oldNumReady)
                ++newNumReady;
            newRecycled.push_back(std::move(s));
        }
        else
        {
            mOverflowSlots.push_back(std::move(s));
        }
        ++idx;
    }
    mRecycledSlots = std::move(newRecycled);
    mNumReadyRecycledSlots = newNumReady;
    mTargetCapacity = newNumSlots;
    TLLM_CHECK_DEBUG(check());
}

bool SlotAllocator::finishShrink()
{
    TLLM_CHECK_DEBUG(check());
    // Overflow-range IDs that were ever issued are exactly
    // max(0, _num_active_slots - _target_capacity); the underused case
    // (_num_active_slots <= _target_capacity) collapses to zero.
    SlotCount const expectedOverflow = std::max(SlotCount{0}, mNumActiveSlots - mTargetCapacity);
    if (shrinkInProgress() && slotCountValueFromSize(mOverflowSlots.size()) == expectedOverflow)
    {
        // Validate uniqueness of slot IDs in overflow (debug only).
        if (TLLM_UNLIKELY(gDebug))
        {
            std::set<SlotId> ids;
            for (auto const& s : mOverflowSlots)
            {
                TLLM_CHECK(s.hasValidSlot());
                ids.insert(s.slotId());
            }
            TLLM_CHECK_WITH_INFO(ids.size() == mOverflowSlots.size(), "Duplicate slot IDs in overflow slots");
        }
        // Synchronize overflow events. Slots often share an event; synchronize() closes it, so
        // later slots holding the same event return immediately.
        for (auto& s : mOverflowSlots)
            s.readyEvent.synchronize();
        for (auto& s : mOverflowSlots)
            s.resetSlot();
        mOverflowSlots.clear();
        mCapacity = mTargetCapacity;
        mNumActiveSlots = std::min(mNumActiveSlots, mCapacity);
        scrubEvents();
        TLLM_CHECK_DEBUG(check());
        return true;
    }
    throw std::runtime_error("SlotAllocator::finishShrink: cannot finish shrink yet");
}

std::vector<SlotId> SlotAllocator::getSlotsBlockingShrink() const
{
    std::vector<SlotId> result;
    for (SlotCount id = mTargetCapacity; id < mCapacity; ++id)
    {
        if (mOccupiedMask.get(slotCountToSizeT(id)))
            result.push_back(SlotId{id});
    }
    return result;
}

void SlotAllocator::synchronize()
{
    while (slotCountToSizeT(mNumReadyRecycledSlots) != mRecycledSlots.size())
    {
        scrubEvents();
    }
}

bool SlotAllocator::check() const noexcept
{
    // Mirrors Python SlotAllocator._check().
    if (mNumActiveSlots > mCapacity)
        return false;
    if (mTargetCapacity > mCapacity)
        return false;
    if (!shrinkInProgress() && !mOverflowSlots.empty())
        return false;
    for (auto const& slot : mOverflowSlots)
    {
        if (!slot.hasValidSlot())
            return false;
        SlotCount id = slot.slotId().value();
        if (id < mTargetCapacity || id >= mCapacity)
            return false;
    }
    SlotCount const accountedSlots = slotCountValueFromSize(mRecycledSlots.size())
        + slotCountValueFromSize(mOverflowSlots.size()) + numOccupiedSlots();
    if (accountedSlots != mNumActiveSlots)
        return false;
    return true;
}

void SlotAllocator::scrubEvents()
{
    for (size_t i = slotCountToSizeT(mNumReadyRecycledSlots); i < mRecycledSlots.size(); ++i)
    {
        if (mRecycledSlots[i].queryReady())
        {
            ++mNumReadyRecycledSlots;
        }
        else
        {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// GpuSlotPool
// ---------------------------------------------------------------------------

GpuSlotPool::GpuSlotPool(size_t slotSize, size_t vmSize, PooledPhysMemAllocator& physMemAllocator, SlotCount numSlots)
    : SlotPoolBase(slotSize)
    , mVirtMem(vmSize, physMemAllocator)
{
    TLLM_CHECK_DEBUG_WITH_INFO(
        vmSize % physMemAllocator.physMemSize() == 0, "vm_size must be aligned to phys_mem_size");
    resize(numSlots);
}

size_t GpuSlotPool::computeNumPhysMem(size_t slotSize, SlotCount numSlots, size_t physMemSize) noexcept
{
    return divUp(slotCountToSizeT(numSlots) * slotSize, physMemSize);
}

SlotCount GpuSlotPool::computeNumSlots(size_t slotSize, size_t numPhysMem, size_t physMemSize) noexcept
{
    return slotCountValueFromSize(numPhysMem * physMemSize / slotSize);
}

SlotCount GpuSlotPool::numSlots() const noexcept
{
    return computeNumSlots(mSlotSize, mVirtMem.numPhysMem(), mVirtMem.physMemSize());
}

void GpuSlotPool::destroy()
{
    mVirtMem.destroy();
}

void GpuSlotPool::resize(SlotCount newNumSlots)
{
    size_t physSize = mVirtMem.physMemSize();
    size_t newPhysMem = computeNumPhysMem(mSlotSize, newNumSlots, physSize);
    mVirtMem.realloc(physSize * newPhysMem);
}

SlotCount GpuSlotPool::extendByOnePhysMem()
{
    mVirtMem.extend(1);
    return numSlots();
}

Address GpuSlotPool::slotAddress(SlotId slot) const
{
    TLLM_CHECK_DEBUG_WITH_INFO(isValidSlotId(slot), "GpuSlotPool::slotAddress: slot index out of bounds");
    return MemAddress(mVirtMem.address() + mSlotSize * toSizeT(slot));
}

// ---------------------------------------------------------------------------
// HostSlotPool
// ---------------------------------------------------------------------------

HostSlotPool::HostSlotPool(size_t slotSize, SlotCount numSlots)
    : SlotPoolBase(slotSize)
    , mHostMem(alignedSize(numSlots))
{
}

size_t HostSlotPool::alignedSize(SlotCount numSlots) const noexcept
{
    return roundUp(slotCountToSizeT(numSlots) * mSlotSize, HostMem::kAlignment);
}

SlotCount HostSlotPool::numSlots() const noexcept
{
    return slotCountValueFromSize(mHostMem.size() / mSlotSize);
}

void HostSlotPool::destroy()
{
    mHostMem.destroy();
}

void HostSlotPool::resize(SlotCount newNumSlots)
{
    mHostMem.resize(alignedSize(newNumSlots));
}

Address HostSlotPool::slotAddress(SlotId slot) const
{
    TLLM_CHECK_DEBUG_WITH_INFO(isValidSlotId(slot), "HostSlotPool::slotAddress: slot index out of bounds");
    return MemAddress(mHostMem.address() + mSlotSize * toSizeT(slot));
}

// ---------------------------------------------------------------------------
// DiskSlotPool
// ---------------------------------------------------------------------------

DiskSlotPool::DiskSlotPool(std::string const& directory, size_t slotSize, SlotCount numSlots)
    : SlotPoolBase(slotSize)
{
    // Try O_TMPFILE first, fall back to mkstemp.
    mFd = ::open(directory.c_str(), O_TMPFILE | O_RDWR | O_EXCL, 0664);
    if (mFd < 0)
    {
        if (errno == EOPNOTSUPP)
        {
            char tmpl[PATH_MAX];
            snprintf(tmpl, sizeof(tmpl), "%s/kvXXXXXX", directory.c_str());
            mFd = ::mkstemp(tmpl);
            if (mFd < 0)
            {
                throw DiskOOMError("DiskSlotPool: mkstemp failed: " + std::string(std::strerror(errno)));
            }
            ::unlink(tmpl);
        }
        else
        {
            throw DiskOOMError("DiskSlotPool: open O_TMPFILE failed: " + std::string(std::strerror(errno)));
        }
    }
    resize(numSlots);
}

DiskSlotPool::~DiskSlotPool()
{
    destroy();
}

SlotCount DiskSlotPool::numSlots() const noexcept
{
    return mNumSlots;
}

void DiskSlotPool::destroy()
{
    if (mFd != kBadFileDescriptor)
    {
        ::close(mFd);
        mFd = kBadFileDescriptor;
    }
    mNumSlots = SlotCount{0};
}

void DiskSlotPool::resize(SlotCount newNumSlots)
{
    // Recorded only once the file really changed size: resizeFile throws without resizing.
    resizeFile(mFd, slotCountToSizeT(newNumSlots) * mSlotSize);
    mNumSlots = newNumSlots;
}

Address DiskSlotPool::slotAddress(SlotId slot) const
{
    TLLM_CHECK_WITH_INFO(isValidSlotId(slot), "DiskSlotPool::slotAddress: slot index out of bounds");
    size_t const byteOffset = toSizeT(slot) * mSlotSize;
    TLLM_CHECK_DEBUG_WITH_INFO(byteOffset <= static_cast<size_t>(std::numeric_limits<ssize_t>::max()),
        "DiskSlotPool::slotAddress: byte offset out of range");
    return DiskAddress{mFd, static_cast<ssize_t>(byteOffset)};
}

// ---------------------------------------------------------------------------
// PoolGroupBase
// ---------------------------------------------------------------------------

PoolGroupBase::PoolGroupBase(SlotCount numSlots)
    : mSlotAllocator(numSlots)
{
}

SlotCount PoolGroupBase::getNumSlotsFromPools() const noexcept
{
    if (mPools.empty())
        return 0;
    SlotCount minSlots = mPools.front()->numSlots();
    for (PoolIndex poolIdx{1}; poolIdx < mPools.size(); ++poolIdx)
    {
        minSlots = std::min(minSlots, mPools[poolIdx]->numSlots());
    }
    return minSlots;
}

PoolGroupBase::~PoolGroupBase()
{
    KVCM2_POISON_ON_EXCEPT([this]() { destroy(); });
}

SlotCount PoolGroupBase::numSlots() const noexcept
{
    SlotCount n = mSlotAllocator.numSlots();
    if (TLLM_UNLIKELY(gDebug))
    {
        // Mirrors Python PoolGroupBase.num_slots: assert num_slots <= self._get_num_slots_from_pools()
        [[maybe_unused]] SlotCount poolSlots = getNumSlotsFromPools();
        TLLM_CHECK_WITH_INFO(n <= poolSlots, "SlotAllocator capacity exceeds pool capacity");
    }
    return n;
}

Slot PoolGroupBase::allocate()
{
    return mSlotAllocator.allocate();
}

std::vector<Slot> PoolGroupBase::allocateMultiple(SlotCount numSlots)
{
    return mSlotAllocator.allocateMultiple(numSlots);
}

void PoolGroupBase::release(Slot slot)
{
    mSlotAllocator.release(std::move(slot));
}

void PoolGroupBase::destroy()
{
    if (mDestroyed)
        return;
    if (mSlotAllocator.numSlots() != 0)
    {
        mSlotAllocator.synchronize();
        mSlotAllocator.prepareForShrink(0);
        mSlotAllocator.finishShrink();
    }
    for (auto& p : mPools)
        p->destroy();
    mDestroyed = true;
}

void PoolGroupBase::resizePools(std::optional<SlotCount> newNumSlots)
{
    SlotCount n = newNumSlots.value_or(mSlotAllocator.numSlots());
    for (auto& p : mPools)
        p->resize(n);
    // Mirrors Python PoolGroupBase.resize_pools: assert NDEBUG or self._check(True)
    // After resize, allocator capacity must not exceed pool capacity (allow mismatch).
    TLLM_CHECK_DEBUG_WITH_INFO(mSlotAllocator.numSlots() <= getNumSlotsFromPools(),
        "After resizePools: allocator capacity exceeds pool capacity");
}

TypedVec<PoolIndex, Address> PoolGroupBase::slotAddress(SlotId slotId) const
{
    TypedVec<PoolIndex, Address> addrs;
    addrs.reserve(mPools.size());
    for (auto const& p : mPools)
        addrs.push_back(p->slotAddress(slotId));
    return addrs;
}

TypedVec<PoolIndex, size_t> PoolGroupBase::slotSize() const
{
    TypedVec<PoolIndex, size_t> sizes;
    sizes.reserve(mPools.size());
    for (auto const& p : mPools)
        sizes.push_back(p->slotSize());
    return sizes;
}

// ---------------------------------------------------------------------------
// GpuPoolGroup
// ---------------------------------------------------------------------------

GpuPoolGroup::GpuPoolGroup(SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList,
    PooledPhysMemAllocator& physMemAllocator, TypedVec<PoolIndex, bool> sparsity)
    : PoolGroupBase(numSlots)
    , mSparsity(std::move(sparsity))
{
    if (mSparsity.empty())
    {
        mSparsity = TypedVec<PoolIndex, bool>(slotSizeList.size(), false);
    }
    TLLM_CHECK_WITH_INFO(
        mSparsity.size() == slotSizeList.size(), "GpuPoolGroup: one sparsity flag per pool is required");
    size_t physMemSize = physMemAllocator.physMemSize();
    // Query total GPU memory to size virtual address space (mirrors Python GpuPoolGroup).
    size_t totalGpuMem = 0;
    {
        CUdevice dev{};
        cuCheck(cuCtxGetDevice(&dev));
        cuCheck(cuDeviceTotalMem(&totalGpuMem, dev));
    }
    // @TODO: We should replace maxSlotSize with sum. This should also be updated in Python. Will do it later.
    // @TODO: maxSlotSize spans sparse pools too, though they reserve no device VA. When a
    // sparse pool is the largest, every dense pool's vmSize is scaled down by that ratio,
    // which costs growth headroom (the floor below keeps the initial allocation correct),
    // so expandPoolGroup can hit the VA limit earlier than intended. Size the ratio from
    // the device-backed pools alone.
    TLLM_CHECK_WITH_INFO(!slotSizeList.empty(), "GpuPoolGroup: slotSizeList must not be empty");
    size_t maxSlotSize = *std::max_element(slotSizeList.begin(), slotSizeList.end());
    for (PoolIndex p{0}; p < slotSizeList.size(); ++p)
    {
        size_t const sz = slotSizeList.at(p);
        if (isSparse(p))
        {
            // A sparse pool's pages: host memory, device-mapped, so the sparse cache fills
            // from them without a host round trip. No device VA is reserved for them.
            mPools.push_back(std::make_unique<HostSlotPool>(sz, numSlots));
            continue;
        }
        // VA proportional to GPU memory, scaled by slot size ratio (mirrors Python).
        // Compute ratio as double first to avoid size_t overflow.
        double sizeRatio = static_cast<double>(sz) / static_cast<double>(maxSlotSize);
        size_t vmSize = roundDown(static_cast<size_t>(static_cast<double>(totalGpuMem) * sizeRatio), physMemSize);
        vmSize = std::max(vmSize, roundUp(slotCountToSizeT(numSlots) * sz, physMemSize));
        mPools.push_back(std::make_unique<GpuSlotPool>(sz, vmSize, physMemAllocator, numSlots));
    }
}

HostMem const* GpuPoolGroup::hostMem(PoolIndex poolIndex) const
{
    if (!isSparse(poolIndex))
    {
        return nullptr;
    }
    return static_cast<HostSlotPool const&>(*mPools.at(poolIndex)).hostMem();
}

// ---------------------------------------------------------------------------
// HostPoolGroup
// ---------------------------------------------------------------------------

HostPoolGroup::HostPoolGroup(SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList)
    : PoolGroupBase(numSlots)
{
    for (size_t sz : slotSizeList)
    {
        mPools.push_back(std::make_unique<HostSlotPool>(sz, numSlots));
    }
}

HostMem const* HostPoolGroup::hostMem(PoolIndex poolIndex) const
{
    return static_cast<HostSlotPool const&>(*mPools.at(poolIndex)).hostMem();
}

// ---------------------------------------------------------------------------
// DiskPoolGroup
// ---------------------------------------------------------------------------

DiskPoolGroup::DiskPoolGroup(
    SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList, std::string const& directory)
    : PoolGroupBase(numSlots)
{
    for (size_t sz : slotSizeList)
    {
        mPools.push_back(std::make_unique<DiskSlotPool>(directory, sz, numSlots));
    }
}

// ---------------------------------------------------------------------------
// CacheLevelStorage
// ---------------------------------------------------------------------------

std::vector<Slot> CacheLevelStorage::allocateMultiple(PoolGroupIndex pgIdx, SlotCount numSlots)
{
    return mPoolGroups.at(pgIdx)->allocateMultiple(numSlots);
}

void CacheLevelStorage::release(PoolGroupIndex pgIdx, Slot slot)
{
    mPoolGroups.at(pgIdx)->release(std::move(slot));
}

SlotCount CacheLevelStorage::numFreeSlots(PoolGroupIndex pgIdx) const
{
    return mPoolGroups.at(pgIdx)->numFreeSlots();
}

TypedVec<PoolIndex, Address> CacheLevelStorage::slotAddress(PoolGroupIndex pgIdx, SlotId slotId) const
{
    return mPoolGroups.at(pgIdx)->slotAddress(slotId);
}

// ---------------------------------------------------------------------------
// GpuCacheLevelStorage
// ---------------------------------------------------------------------------

GpuCacheLevelStorage::GpuCacheLevelStorage(TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList,
    TypedVec<PoolGroupIndex, SlotCount> const& slotCountList, PooledPhysMemAllocator& physMemAllocator)
    : mPhysMemAllocator(physMemAllocator)
{
    TLLM_CHECK_WITH_INFO(slotCountList.size() == slotDescList.size(),
        "GpuCacheLevelStorage: slotCountList and slotDescList must have the same length");
    for (PoolGroupIndex pgIdx{0}; pgIdx < slotDescList.size(); ++pgIdx)
    {
        auto const& slotDesc = slotDescList[pgIdx];
        auto const sizes = slotDesc.slotSizeList();
        TypedVec<PoolIndex, bool> sparsity;
        sparsity.reserve(sizes.size());
        for (PoolIndex pool{0}; pool < sizes.size(); ++pool)
        {
            sparsity.push_back(slotDesc.isSparse(pool));
        }
        mPoolGroups.push_back(
            std::make_unique<GpuPoolGroup>(slotCountList[pgIdx], sizes, mPhysMemAllocator, std::move(sparsity)));
    }
}

// ---------------------------------------------------------------------------
// HostCacheLevelStorage
// ---------------------------------------------------------------------------

HostCacheLevelStorage::HostCacheLevelStorage(
    TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList, TypedVec<PoolGroupIndex, SlotCount> const& slotCountList)
{
    TLLM_CHECK_WITH_INFO(slotCountList.size() == slotDescList.size(),
        "HostCacheLevelStorage: slotCountList and slotDescList must have the same length");
    for (PoolGroupIndex pgIdx{0}; pgIdx < slotDescList.size(); ++pgIdx)
    {
        mPoolGroups.push_back(
            std::make_unique<HostPoolGroup>(slotCountList[pgIdx], slotDescList[pgIdx].slotSizeList()));
    }
}

// ---------------------------------------------------------------------------
// DiskCacheLevelStorage
// ---------------------------------------------------------------------------

DiskCacheLevelStorage::DiskCacheLevelStorage(TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList,
    TypedVec<PoolGroupIndex, SlotCount> const& slotCountList, std::string directory)
    : mDirectory(std::move(directory))
{
    TLLM_CHECK_WITH_INFO(slotCountList.size() == slotDescList.size(),
        "DiskCacheLevelStorage: slotCountList and slotDescList must have the same length");
    for (PoolGroupIndex pgIdx{0}; pgIdx < slotDescList.size(); ++pgIdx)
    {
        mPoolGroups.push_back(
            std::make_unique<DiskPoolGroup>(slotCountList[pgIdx], slotDescList[pgIdx].slotSizeList(), mDirectory));
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<CacheLevelStorage> createCacheLevelStorage(CacheTierConfig const& tierCfg,
    TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList, TypedVec<PoolGroupIndex, SlotCount> const& slotCountList,
    PooledPhysMemAllocator* gpuPhysMemAllocator)
{
    TLLM_CHECK((cacheTierOf(tierCfg) == CacheTier::GPU_MEM) == (gpuPhysMemAllocator != nullptr));
    return std::visit(
        [&](auto const& cfg) -> std::unique_ptr<CacheLevelStorage>
        {
            using T = std::decay_t<decltype(cfg)>;
            if constexpr (std::is_same_v<T, GpuCacheTierConfig>)
            {
                return std::make_unique<GpuCacheLevelStorage>(slotDescList, slotCountList, *gpuPhysMemAllocator);
            }
            else if constexpr (std::is_same_v<T, HostCacheTierConfig>)
            {
                return std::make_unique<HostCacheLevelStorage>(slotDescList, slotCountList);
            }
            else
            {
                return std::make_unique<DiskCacheLevelStorage>(slotDescList, slotCountList, cfg.path);
            }
        },
        tierCfg);
}

// ---------------------------------------------------------------------------
// CacheLevelStorage helper: grainsToSlots
// Distribute grains among pools within a pool group.
// Returns {num_slots, grains_consumed}.
// Mirrors Python CacheLevelStorage._grains_to_slots.
// ---------------------------------------------------------------------------

std::pair<SlotCount, size_t> CacheLevelStorage::grainsToSlots(size_t pgGrains,
    TypedVec<PoolIndex, size_t> const& slotSizeList, std::vector<PoolIndex> const& which, size_t granularity)
{
    TLLM_CHECK_WITH_INFO(granularity > 0, "Cache storage granularity must be positive");
    // With nothing to divide the grains between there is no slot count to find, and the
    // search below would run past SlotCount's range looking for one.
    TLLM_CHECK_WITH_INFO(!which.empty(), "grainsToSlots needs at least one pool to size against");
    // Indexed by the real pool index, so only the named entries are ever read; the rest
    // stay zero and cost one word each.
    TypedVec<PoolIndex, size_t> minPoolGrains(slotSizeList.size(), 0);
    size_t minTotal = 0;
    for (PoolIndex poolIdx : which)
    {
        TLLM_CHECK_WITH_INFO(slotSizeList[poolIdx] > 0, "Cache slot size must be positive");
        minPoolGrains[poolIdx] = divUp(slotSizeList[poolIdx], granularity);
        minTotal += minPoolGrains[poolIdx];
    }
    if (pgGrains < minTotal)
        return {0, 0};

    SlotCount numSlots{std::numeric_limits<SlotCount>::max()};
    size_t remainingPgGrains = pgGrains;

    // Sort the named pools by slot size ascending. Ties may fall either way: every term
    // below is a function of the sizes alone, so an order that differs only among equal
    // sizes produces the same result.
    std::vector<PoolIndex> poolOrder = which;
    std::sort(poolOrder.begin(), poolOrder.end(),
        [&](PoolIndex a, PoolIndex b) { return slotSizeList[a] < slotSizeList[b]; });

    for (size_t j = 0; j < poolOrder.size(); ++j)
    {
        PoolIndex const poolIdx = poolOrder[j];
        size_t slotSz = slotSizeList[poolIdx];
        size_t poolSzSum = 0;
        for (size_t k = j; k < poolOrder.size(); ++k)
            poolSzSum += slotSizeList[poolOrder[k]];
        double poolFrac = (poolSzSum > 0) ? static_cast<double>(slotSz) / static_cast<double>(poolSzSum) : 1.0;
        size_t roundedGrains = static_cast<size_t>(std::nearbyint(static_cast<double>(remainingPgGrains) * poolFrac));
        size_t poolGrains = std::max(minPoolGrains[poolIdx], roundedGrains);
        SlotCount const poolSlots = slotCountValueFromSize(poolGrains * granularity / slotSz);
        numSlots = std::min(numSlots, poolSlots);
        TLLM_CHECK_DEBUG(poolGrains <= remainingPgGrains);
        remainingPgGrains -= poolGrains;
    }
    TLLM_CHECK_DEBUG(remainingPgGrains == 0);
    TLLM_CHECK_DEBUG(numSlots > 0);

    auto slotsToGrains = [&](SlotCount slots) { return grainsForSlots(slots, slotSizeList, which, granularity); };
    SlotCount lo = numSlots;
    SlotCount step = 1;
    SlotCount hi = lo + step;
    while (slotsToGrains(hi) <= pgGrains)
    {
        lo = hi;
        step *= 2;
        hi = lo + step;
    }
    while (lo + 1 < hi)
    {
        SlotCount const mid = lo + ((hi - lo) / 2);
        if (slotsToGrains(mid) <= pgGrains)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    size_t const used = slotsToGrains(lo);
    TLLM_CHECK_DEBUG(used <= pgGrains);
    TLLM_CHECK_DEBUG(slotsToGrains(lo + 1) > pgGrains);
    return {lo, used};
}

// ---------------------------------------------------------------------------
// CacheLevelStorage helper: grainsForSlots
// Compute minimum grains needed for numSlots in a pool group.
// Mirrors Python CacheLevelStorage._grains_for_slots.
// ---------------------------------------------------------------------------

size_t CacheLevelStorage::grainsForSlots(SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList,
    std::vector<PoolIndex> const& which, size_t granularity)
{
    size_t total = 0;
    for (PoolIndex poolIdx : which)
        total += divUp(slotCountToSizeT(numSlots) * slotSizeList[poolIdx], granularity);
    return total;
}

// ---------------------------------------------------------------------------
// CacheLevelStorage::ratioToSlotCountList (static)
// Mirrors Python CacheLevelStorage.ratio_to_slot_count_list.
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, std::vector<PoolIndex>> CacheLevelStorage::poolsPaidByQuota(
    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& slotSizeLists,
    std::optional<SecondaryPoolBudget> const& secondBudget)
{
    TypedVec<PoolGroupIndex, std::vector<PoolIndex>> ret;
    ret.reserve(slotSizeLists.size());
    for (PoolGroupIndex pgIdx{0}; pgIdx < slotSizeLists.size(); ++pgIdx)
    {
        auto const numPools = slotSizeLists[pgIdx].size();
        TypedVec<PoolIndex, bool> taken(numPools, false);
        if (secondBudget.has_value())
        {
            for (PoolIndex poolIdx : secondBudget->pools[pgIdx])
            {
                TLLM_CHECK_WITH_INFO(poolIdx < numPools,
                    "A second budget names pool %d of group %d, which has only %d pools", poolIdx.value(),
                    pgIdx.value(), numPools.value());
                // Naming a pool twice would charge it twice and leave the quota paying for
                // one fewer than it thinks.
                TLLM_CHECK_WITH_INFO(
                    !taken[poolIdx], "A second budget names pool %d of group %d twice", poolIdx.value(), pgIdx.value());
                taken[poolIdx] = true;
            }
        }
        std::vector<PoolIndex> paid;
        paid.reserve(toSizeT(numPools));
        for (PoolIndex poolIdx{0}; poolIdx < numPools; ++poolIdx)
        {
            if (!taken[poolIdx])
            {
                paid.push_back(poolIdx);
            }
        }
        ret.push_back(std::move(paid));
    }
    return ret;
}

namespace
{

//! Bytes one pool group holds at `slots`, each pool rounded to the grain of the medium it
//! lives in. This is the quantity the pool ratio speaks in: what the group actually
//! occupies, device and host together.
size_t allBytesForSlots(TypedVec<PoolIndex, size_t> const& sizes, std::vector<PoolIndex> const& second, size_t grain,
    size_t secondGrain, SlotCount slots)
{
    auto const n = slotCountToSizeT(slots);
    size_t total = 0;
    for (PoolIndex poolIdx{0}; poolIdx < sizes.size(); ++poolIdx)
    {
        bool const isSecond = std::ranges::find(second, poolIdx) != second.end();
        auto const g = isSecond ? secondGrain : grain;
        total += roundUp(n * sizes[poolIdx], g);
    }
    return total;
}

//! The most slots whose all-bytes cost fits `budget`.
//!
//! Bracketed, not galloped: `A(n) >= n * S` with `S` the summed slot size, so no count above
//! `budget / S` can fit and bisection has an upper end without searching for one. Every pool
//! of the group is in view, so there is no empty-selector case to guard.
SlotCount maxSlotsForAllBytes(TypedVec<PoolIndex, size_t> const& sizes, std::vector<PoolIndex> const& second,
    size_t grain, size_t secondGrain, size_t budget)
{
    size_t sumSizes = 0;
    for (PoolIndex poolIdx{0}; poolIdx < sizes.size(); ++poolIdx)
    {
        sumSizes += sizes[poolIdx];
    }
    TLLM_CHECK_WITH_INFO(sumSizes > 0, "A pool group must have a positive slot size");
    SlotCount lo{0};
    auto hi = slotCountValueFromSize(budget / sumSizes + 1);
    if (allBytesForSlots(sizes, second, grain, secondGrain, hi) <= budget)
    {
        return hi;
    }
    while (lo + 1 < hi)
    {
        SlotCount const mid = lo + ((hi - lo) / 2);
        if (allBytesForSlots(sizes, second, grain, secondGrain, mid) <= budget)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

//! The most slots the grains already bought for `slots` will hold, in every pool at once.
//!
//! The counterpart of PoolGroupBase::getNumSlotsFromPools, and what makes a count
//! self-consistent: growing to it costs nothing in any medium, so it never disturbs a
//! budget. It replaces the tail bump loop the single-budget allocator used to run.
//!
//! Every pool is in view, including ones the second budget pays for and whether or not that
//! budget has a quota. A host-backed pool nobody is bounding is still real memory: dropping
//! it here would let a count grow to whatever the device grain absorbs and allocate host rows
//! for all of it, which is the opposite of what the quota exists to do.
SlotCount freeRun(TypedVec<PoolIndex, size_t> const& sizes, std::vector<PoolIndex> const& second, size_t grain,
    size_t secondGrain, SlotCount slots)
{
    auto const n = slotCountToSizeT(slots);
    SlotCount best{std::numeric_limits<SlotCount>::max()};
    for (PoolIndex poolIdx{0}; poolIdx < sizes.size(); ++poolIdx)
    {
        bool const isSecond = std::ranges::find(second, poolIdx) != second.end();
        auto const g = isSecond ? secondGrain : grain;
        auto const sz = sizes[poolIdx];
        best = std::min(best, slotCountValueFromSize(divUp(n * sz, g) * g / sz));
    }
    return best;
}

} // namespace

TypedVec<PoolGroupIndex, SlotCount> CacheLevelStorage::ratioToSlotCountList(size_t quota,
    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& slotSizeLists,
    TypedVec<PoolGroupIndex, float> const& ratioList, size_t granularity,
    TypedVec<PoolGroupIndex, SlotCount> const& minSlots, std::optional<SecondaryPoolBudget> const& secondBudget)
{
    PoolGroupIndex const numPg = slotSizeLists.size();
    TLLM_CHECK_DEBUG(ratioList.size() == numPg);
    TLLM_CHECK_WITH_INFO(granularity > 0, "Cache storage granularity must be positive");

    bool const hasSecond = secondBudget.has_value();
    bool const bounded = hasSecond && secondBudget->quota.has_value();
    size_t const grain2 = hasSecond ? secondBudget->granularity : granularity;
    TLLM_CHECK_WITH_INFO(!hasSecond || grain2 > 0, "A second budget needs a positive granularity");

    // The quota pays for whatever the second budget does not name, so the two partition the
    // group's pools by construction rather than by agreement between callers.
    auto const pools = poolsPaidByQuota(slotSizeLists, secondBudget);
    static std::vector<PoolIndex> const kNoPools;
    auto secondPools = [&](PoolGroupIndex pgIdx) -> std::vector<PoolIndex> const&
    { return hasSecond ? secondBudget->pools[pgIdx] : kNoPools; };

    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        TLLM_CHECK_WITH_INFO(ratioList[pgIdx] > 0.f, "ratioToSlotCountList: all ratios must be positive");
        TLLM_CHECK_WITH_INFO(!pools[pgIdx].empty(),
            "ratioToSlotCountList: pool group %d has no pool the primary quota pays for", pgIdx.value());
    }

    auto quotaGrains = [&](PoolGroupIndex pgIdx, SlotCount slots)
    { return grainsForSlots(slots, slotSizeLists[pgIdx], pools[pgIdx], granularity); };
    auto secondGrains = [&](PoolGroupIndex pgIdx, SlotCount slots)
    { return grainsForSlots(slots, slotSizeLists[pgIdx], secondPools(pgIdx), grain2); };
    // What the ratio speaks in: every pool, each in the grain of its own medium.
    auto allBytes = [&](PoolGroupIndex pgIdx, SlotCount slots)
    { return allBytesForSlots(slotSizeLists[pgIdx], secondPools(pgIdx), granularity, grain2, slots); };

    size_t const budgetGrains = quota / granularity;
    size_t const budget2Grains = bounded ? *secondBudget->quota / grain2 : 0;

    // The floors are not negotiable, so they come off the top before anything is shared.
    // Snapping each to its free run costs no grain in any medium, and starting there is what
    // makes "never decrease" enough to keep them.
    TypedVec<PoolGroupIndex, SlotCount> slotCntList(numPg, 0);
    size_t floorQuota = 0;
    size_t floorSecond = 0;
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        slotCntList[pgIdx] = freeRun(
            slotSizeLists[pgIdx], secondPools(pgIdx), granularity, grain2, std::max(minSlots[pgIdx], SlotCount{0}));
        floorQuota += quotaGrains(pgIdx, slotCntList[pgIdx]);
        floorSecond += secondGrains(pgIdx, slotCntList[pgIdx]);
    }
    if (bounded)
    {
        // The only way the two hard bounds can conflict, and it names neither the ratio nor
        // the quota -- so it is a property of the configuration, caught the first time this
        // runs and unable to begin failing at a later rebalance.
        TLLM_CHECK_WITH_INFO(floorSecond <= budget2Grains,
            "Second memory budget of %zu bytes cannot hold the %zu the batches in `constraints` need. Raise the "
            "budget or lower the constraints.",
            *secondBudget->quota, floorSecond * grain2);
    }
    TLLM_CHECK_WITH_INFO(floorQuota <= budgetGrains, "Insufficient quota to satisfy min_slots constraints");

    size_t remainingQuota = budgetGrains - floorQuota;
    size_t remainingSecond = bounded ? budget2Grains - floorSecond : 0;

    // One demand curve, driven by the combined all-bytes budget divided by the ratio. A
    // group's demand at level `x` is what `x * r` buys of its own all-bytes cost per slot,
    // so when nothing binds every group sits at the same level -- and equal levels are
    // exactly the requested ratio. The per-medium budgets then act as caps on that demand.
    auto demand = [&](PoolGroupIndex pgIdx, double level)
    {
        auto const share = static_cast<size_t>(std::max(0.0, level * static_cast<double>(ratioList[pgIdx])));
        auto const want = maxSlotsForAllBytes(slotSizeLists[pgIdx], secondPools(pgIdx), granularity, grain2, share);
        return std::max(slotCntList[pgIdx], want);
    };

    // A level wide enough that every group's demand is capped by a budget rather than by it.
    double maxLevel = 0.0;
    {
        double minRatio = 1.0;
        for (auto r : ratioList)
        {
            minRatio = std::min(minRatio, static_cast<double>(r));
        }
        auto const totalBytes = static_cast<double>(budgetGrains) * static_cast<double>(granularity)
            + static_cast<double>(budget2Grains) * static_cast<double>(grain2);
        maxLevel = (totalBytes + 1.0) / std::max(minRatio, 1e-12);
    }

    std::vector<PoolGroupIndex> active;
    active.reserve(toSizeT(numPg));
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        active.push_back(pgIdx);
    }

    // Raise the level until one budget cannot pay for the next step, freezing the groups the
    // second budget holds down and returning what they do not spend of the first.
    //
    // Each round either settles everyone and breaks, or freezes a non-empty `held` and drops it
    // from `active`. So `active` strictly shrinks -- at most one round per pool group, with no
    // round that changes nothing. Nothing re-enters `active` and neither remainder grows, so the
    // search cannot oscillate between two levels.
    while (!active.empty())
    {
        // How far the second budget alone would let the groups it pays for rise.
        double levelSecond = maxLevel;
        if (bounded)
        {
            bool anyOnSecond = false;
            for (PoolGroupIndex pgIdx : active)
            {
                anyOnSecond = anyOnSecond || !secondPools(pgIdx).empty();
            }
            if (anyOnSecond)
            {
                double lo = 0.0;
                double hi = maxLevel;
                for (int step = 0; step < 64; ++step)
                {
                    double const mid = lo + (hi - lo) / 2.0;
                    size_t spent = 0;
                    for (PoolGroupIndex pgIdx : active)
                    {
                        if (!secondPools(pgIdx).empty())
                        {
                            spent += secondGrains(pgIdx, demand(pgIdx, mid)) - secondGrains(pgIdx, slotCntList[pgIdx]);
                        }
                    }
                    (spent <= remainingSecond ? lo : hi) = mid;
                }
                levelSecond = lo;
            }
        }

        auto capped = [&](PoolGroupIndex pgIdx, double level)
        { return demand(pgIdx, secondPools(pgIdx).empty() ? level : std::min(level, levelSecond)); };

        // How far the quota lets everyone rise, with the second budget's cap already applied.
        double levelQuota = 0.0;
        {
            double lo = 0.0;
            double hi = maxLevel;
            for (int step = 0; step < 64; ++step)
            {
                double const mid = lo + (hi - lo) / 2.0;
                size_t spent = 0;
                for (PoolGroupIndex pgIdx : active)
                {
                    spent += quotaGrains(pgIdx, capped(pgIdx, mid)) - quotaGrains(pgIdx, slotCntList[pgIdx]);
                }
                (spent <= remainingQuota ? lo : hi) = mid;
            }
            levelQuota = lo;
        }

        std::vector<PoolGroupIndex> held;
        for (PoolGroupIndex pgIdx : active)
        {
            if (!secondPools(pgIdx).empty() && demand(pgIdx, levelSecond) < demand(pgIdx, levelQuota))
            {
                held.push_back(pgIdx);
            }
        }

        if (held.empty())
        {
            for (PoolGroupIndex pgIdx : active)
            {
                auto const slots = capped(pgIdx, levelQuota);
                remainingQuota -= quotaGrains(pgIdx, slots) - quotaGrains(pgIdx, slotCntList[pgIdx]);
                if (bounded)
                {
                    remainingSecond -= secondGrains(pgIdx, slots) - secondGrains(pgIdx, slotCntList[pgIdx]);
                }
                slotCntList[pgIdx] = slots;
            }
            break;
        }

        // Settle the held groups and hand back the quota grains they will not use, so the
        // groups the second budget does not touch can take them.
        for (PoolGroupIndex pgIdx : held)
        {
            auto const slots = demand(pgIdx, levelSecond);
            remainingQuota -= quotaGrains(pgIdx, slots) - quotaGrains(pgIdx, slotCntList[pgIdx]);
            remainingSecond -= secondGrains(pgIdx, slots) - secondGrains(pgIdx, slotCntList[pgIdx]);
            slotCntList[pgIdx] = slots;
        }
        std::vector<PoolGroupIndex> rest;
        rest.reserve(active.size() - held.size());
        for (PoolGroupIndex pgIdx : active)
        {
            if (std::ranges::find(held, pgIdx) == held.end())
            {
                rest.push_back(pgIdx);
            }
        }
        active = std::move(rest);
    }

    // A single level cannot spend an integral remainder: after it stops, one group's next
    // step is unaffordable while another's still fits. Give each remaining step to whichever
    // group would sit lowest afterwards, which is where the ratio wants it.
    //
    // Each iteration raises exactly one group's count, so there is no zero-change round and no
    // way back to a count already visited. Every step crosses a grain in some medium and quota
    // grains grow without bound in the slot count, so every group eventually prices itself out.
    // The level search has already spent all but a fraction of a step, which is what keeps the
    // count of iterations small rather than merely finite.
    for (;;)
    {
        auto best = numPg;
        double bestLevel = 0.0;
        SlotCount bestSlots{0};
        for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
        {
            // Strictly greater than the current count, so every accepted step is progress and
            // the loop cannot spin.
            auto const next
                = freeRun(slotSizeLists[pgIdx], secondPools(pgIdx), granularity, grain2, slotCntList[pgIdx] + 1);
            TLLM_CHECK_DEBUG(next > slotCntList[pgIdx]);
            auto const dq = quotaGrains(pgIdx, next) - quotaGrains(pgIdx, slotCntList[pgIdx]);
            auto const ds = secondGrains(pgIdx, next) - secondGrains(pgIdx, slotCntList[pgIdx]);
            if (dq > remainingQuota || (bounded && ds > remainingSecond))
            {
                continue;
            }
            double const level = static_cast<double>(allBytes(pgIdx, next)) / static_cast<double>(ratioList[pgIdx]);
            if (best == numPg || level < bestLevel)
            {
                best = pgIdx;
                bestLevel = level;
                bestSlots = next;
            }
        }
        if (best == numPg)
        {
            break;
        }
        remainingQuota -= quotaGrains(best, bestSlots) - quotaGrains(best, slotCntList[best]);
        if (bounded)
        {
            remainingSecond -= secondGrains(best, bestSlots) - secondGrains(best, slotCntList[best]);
        }
        slotCntList[best] = bestSlots;
    }

    return slotCntList;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
