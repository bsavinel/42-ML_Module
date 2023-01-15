from matrix import *

def printListInMatix(string, listM):
	if (isinstance(string, str) and isinstance(listM, list)):
		print(string)
		print("size : (", len(listM), "," , len(listM[0]), ")\nContent :")
		for i in range(len(listM)):
			print(listM[i])
		print("")
	else:
		return NotImplemented
	
print("""############################################
############   Initialisation   ############
############################################\n""")

m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print("m1 :")
print(m1)
m2 = Matrix((3, 2))
print("\nm2 :")
print(m2)
try:
	m3 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0]])
except MatrixException as e:
	print("\nError :", e)
m4 = Vector((1, 3))
print("\nm4 :")
print(m4)
m5 = Vector([[0.0], [1.0], [2.0]])
print("\nm5 :")
print(m5)
m6 = Vector((3 , 1))
print("\nm6 :")
print(m6)
try:
	m7 = Vector((3,2))
except MatrixException as e:
	print("\nError :", e)

print("\n-------   Result expected   -------\n")
printListInMatix("m1 :", [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
printListInMatix("m2 :", [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
print("Error because m3 is not rectangular\n")
printListInMatix("m4 :", [[0.0, 0.0, 0.0]])
printListInMatix("m5 :", [[0.0], [1.0], [2.0]])
printListInMatix("m6 :", [[0.0], [0.0], [0.0]])
print("Error because m7 is not a vector\n")

print("""#####################################
############   Adition   ############
#####################################\n""")

m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
m2 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
m3 = m1 + m2
print("m3 :")
print(m3)
try:
	m4 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
	m5 = m1 + m4
except MatrixException as e:
	print("\nError :", e)

print("\n-------   Result expected   -------\n")

printListInMatix("m3 :", [[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]])
print("Error because m1 and m4 don't have the same dimmension\n")

print("""##########################################
############   substraction   ############
##########################################\n""")

m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
m2 = Matrix([[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]])
m3 = m2 - m1
print("m3 :")
print(m3)
try:
	m4 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
	m5 = m1 - m4
except MatrixException as e:
	print("\nError :", e)

print("\n-------   Result expected   -------\n")

printListInMatix("m3 :", [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print("Error because m1 and m4 don't have the same dimmension\n")

print("""############################################
############   Multiplication   ############
############################################\n""")

m1 = Matrix(([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]]))
m2 = m1 * 2
print("m2 :")
print(m2)
m3 = 2 * m1
print("\nm3 :")
print(m3)
m4 = Matrix([[0.0, 1.0],[2.0, 3.0],[4.0, 5.0],[6.0, 7.0]])
m5 = m1 * m4
print("\nm5 :")
print(m5)

try:
	m6 = Matrix([[0.0, 1.0, 2.0]])
	m7 = m1 * m6
except MatrixException as e:
	print("\nError :", e)

m7 = Matrix([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
m8 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
v1 = Vector([[1], [2], [3]])
v2 = Vector([[1, 2, 3]])
v3 = m7 * v1
v4 = v2 * m8
print("\nv3 :")
print(v3)
print("\nv4 :")
print(v4)

print("\n-------   Result expected   -------\n")

printListInMatix("m2 :", [[0.0, 2.0, 4.0, 6.0], [0.0, 4.0, 8.0, 12.0]])
printListInMatix("m3 :", [[0.0, 2.0, 4.0, 6.0], [0.0, 4.0, 8.0, 12.0]])
printListInMatix("m5 :", [[28., 34.], [56., 68.]])
print("Error because m1 and m6 don't have the compatible dimension\n")
printListInMatix("v3 :", [[8.], [16.]])
printListInMatix("v4 :", [[16., 22.]])

print("""######################################
############   Division   ############
######################################\n""")

m1 = Matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
m2 = m1 / 2
print("m2 :")
print(m2)

print("\n-------   Result expected   -------\n")

printListInMatix("m2 :", [[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]])

print("""###########################################
############   Transposition   ############
###########################################\n""")

m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print("m1 :")
print(m1)
m2 = m1.T()
print("\nm2 :")
print(m2)
m3 = m2.T()
print("\nm3 :")
print(m3)
print("\n-------   Result expected   -------\n")
printListInMatix("m1 :", [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
printListInMatix("m2 :", [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])
printListInMatix("m3 :", [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])