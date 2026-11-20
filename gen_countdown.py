import json
import random

PROMPT = ("Using the numbers {nums}, create an equation that equals {target}. "
          "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
          "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, "
          "for example <answer> (1 + 2) / 3 </answer>.")


def solvable_target(nums, rng):
    while True:
        pool = list(nums)
        while len(pool) > 1:
            a, b = pool.pop(rng.randrange(len(pool))), pool.pop(rng.randrange(len(pool)))
            op = rng.choice(['+', '-', '*', '/'])
            if op == '+':
                r = a + b
            elif op == '-':
                r = a - b
            elif op == '*':
                r = a * b
            else:
                if b == 0 or a % b != 0:
                    break
                r = a // b
            if r <= 0:
                break
            pool.append(r)
        else:
            if 1 <= pool[0] <= 100:
                return pool[0]


def gen(n, seed):
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        k = rng.choice([3, 4])
        nums = [rng.randint(1, 100) for _ in range(k)]
        target = solvable_target(nums, rng)
        if target is None:
            continue
        rows.append({
            'messages': [{'role': 'user', 'content': PROMPT.format(nums=nums, target=target)}],
            'target': target,
            'nums': nums,
        })
    return rows


if __name__ == '__main__':
    for name, n, seed in [('train', 20000, 42), ('test', 256, 7)]:
        with open(f'/data1/yesiliang/tinyzero_swift/data/countdown_{name}.jsonl', 'w') as f:
            for row in gen(n, seed):
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(name, n)
