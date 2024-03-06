import numpy as np
with open("stagbeetle208x208x123.dat", "rb") as file:
    #binary_data = file.read()
    temp_array = np.fromfile(file, dtype=np.uint16)

print(len(temp_array))
temp_array = temp_array[3:]
print(len(temp_array))
lst = temp_array.reshape(208,208,123)
print(lst.shape)
lst.astype('uint16').tofile('4b.raw')