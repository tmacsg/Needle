import gc
import needle as ndl

def check_array_cnt():
    count = 0
    for obj in gc.get_objects():  
        if isinstance(obj, ndl.NDArray):
            count += 1
    return count

def trace_call(func):
    def wrapper(*args, **kwargs):  
        count_1 = check_array_cnt()
        result = func(*args, **kwargs)
        count_2 = check_array_cnt()
        classname = type(args[0]).__name__
        # if 'Broad' in classname:
        print(f'{classname}.{func.__name__}, {count_2} - {count_1} = {count_2 - count_1}')
        return result
    return wrapper