def moving_average(nums, k):
    if k <= 0:
        raise ValueError("k must be positive")
    if not nums:
        return []
    if k > len(nums):
        raise ValueError("k larger than the input")
    return [sum(nums[i : i + k]) / k for i in range(len(nums) - k + 1)]
