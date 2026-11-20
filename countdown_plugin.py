import re

from swift.rewards import ORM, orms


def extract_answer(text):
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


def validate_equation(equation, nums):
    used = sorted(int(n) for n in re.findall(r'\d+', equation))
    return used == sorted(nums)


def evaluate_equation(equation):
    if not re.match(r'^[\d+\-*/().\s]+$', equation):
        return None
    try:
        return eval(equation, {'__builtins__': None}, {})
    except Exception:
        return None


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs):
        scores = []
        for completion, t, n in zip(completions, target, nums):
            equation = extract_answer(completion)
            if equation is None or not validate_equation(equation, n):
                scores.append(0.0 if equation is None else 0.1)
                continue
            result = evaluate_equation(equation)
            if result is None:
                scores.append(0.1)
            elif abs(result - t) < 1e-5:
                scores.append(1.0)
            else:
                scores.append(0.1)
        return scores


orms['countdown'] = CountdownORM
