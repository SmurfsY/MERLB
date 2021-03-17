import torch
import numpy as np
test_a = [1,2,3,4,5,6]
test_b = [2,1,2,1,2,1]
test_c = [a / b for a,b in zip(test_a, test_b)]
print(test_c)