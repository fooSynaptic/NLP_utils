# fastsort alogrithm


def quicksort(arr, p, r):
	if p < r:
		q = my_PARTITION(arr, p, r)
		quicksort(arr, p, q-1)
		quicksort(arr, q+1, r)



def my_PARTITION(arr, p, r):
	x,i = arr[r], p-1
	#i = p-1

	for j in range(p, r):
		if arr[j] <= x:
			i = i + 1
			arr[i], arr[j] = arr[j], arr[i]			
	arr[r], arr[i+1] = arr[i+1], arr[r]
	
	return i+1





arr = [2,8,7,1,3,5,6,4]

quicksort(arr, 0, 7)

print(arr)

arr2 = [2,5,4,6,7,9,10,1]
quicksort(arr2, 0, 7)
print(arr2)

```
[1, 2, 3, 4, 5, 6, 7, 8]
[1, 2, 4, 5, 6, 7, 9, 10]
```
