# build fastsort alogrithm from scratch


def quicksort(arr, p, r):
	if p < r:
		q = my_PARTITION(arr, p, r)
		quicksort(arr, p, q-1)
		quicksort(arr, q+1, r)



def my_PARTITION(arr, p, r):
	x = arr[r-1]
	i = p-1

	for j in range(p, r-1):
		if arr[j] <= x:
			i = i + 1
			m=arr[j] 
			arr[j]=arr[i] 
			arr[i]=m


	s = arr[r-1]
	arr[r-1] = arr[i+1]
	arr[i+1] = s
	
	return i+1





arr = [2,8,7,1,3,5,6,4]

quicksort(arr, 0, 8)

print(arr)
