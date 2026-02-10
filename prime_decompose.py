# 对任意正整数做质因数分解

def prime_decompose(n):
    factors = []
    # 从最小的质数 2 开始尝试
    divisor = 2
    while divisor * divisor <= n:  # 只需要检查到 sqrt(n)
        while n % divisor == 0:  # 如果当前质数是 n 的因子
            factors.append(divisor)  # 将质数添加到结果列表
            n //= divisor  # 将 n 除以这个质数，继续检查是否还有这个因子
        divisor += 1  # 尝试下一个整数作为潜在的质数
    if n > 1:  # 如果最后剩下的 n 大于 1，那么它也是一个质数
        factors.append(n)
    return factors


if __name__ == "__main__":
    number = int(input("请输入一个正整数进行质因数分解: "))
    result = prime_decompose(number)
    print(f"{number} 的质因数分解结果是: {result}")