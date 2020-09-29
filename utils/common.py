from functools import wraps


def partial(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError as e:
            if 'required positional argument' in str(e):
                return partial(
                    lambda *args2, **kwargs2:
                        func(*args, *args2, **kwargs, **kwargs2)
                )
            else:
                raise e

    return wrapper


def pipe(*funcs):
    def wrapped(*args, **kwargs):
        x = funcs[0](*args, **kwargs)
        for f in funcs[1:]:
            x = f(x)
        return x

    return wrapped


if __name__ == '__main__':
    @partial
    def func(a, b, c):
        return a + b + c

    print(func(1, 2)(3))
    print(func(1)(2, 3))
    print(func(1)(2)(3))
    print(func(1, 2, 3))
