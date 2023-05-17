class MatrixException(Exception):
	def __init__(self, message):
		self.message = message
		super().__init__(self.message)

#!####################################################################################################!#
#!#############################################  MATRIX  #############################################!#
#!####################################################################################################!#

class Matrix:

#!----------------Matirx initialisateur----------------!#

	def __init__(self, matrix_data):
		if (isinstance(matrix_data, list)):
			if not all(len(it) == len(matrix_data[0]) for it in matrix_data):
				raise MatrixException("Matrix must be rectangular")
			self.data = matrix_data
			self.shape = (len(matrix_data), len(matrix_data[0]))
		elif (isinstance(matrix_data, tuple) and len(matrix_data) == 2 and matrix_data[0] > 0 and matrix_data[1] > 0):
			self.shape = matrix_data
			self.data = [[0 for i in range(self.shape[1])] for j in range(self.shape[0])]
		else:
			raise MatrixException("Matrix must be created with a list with the content or a tuple with the dimension")

#!----------------Matrix adition----------------!#

	def __add__(self, other):
		if (isinstance(other, Matrix)):
			if ((not isinstance(self, Vector)) and isinstance(other, Vector)):
				return NotImplemented
			if (self.shape != other.shape):
				raise MatrixException("Matrix can be added only if they have the same shape")
			return type(self)([[self.data[j][i] + other.data[j][i] for i in range(self.shape[1])] for j in range(self.shape[0])])
		else:
			raise MatrixException("Matrix can be added only with another matrix")

	def __radd__(self, other):
		if (isinstance(other, Matrix)):
			if ((not isinstance(self, Vector)) and isinstance(other, Vector)):
				return NotImplemented
			if (self.shape != other.shape):
				raise MatrixException("Matrix can be added only if they have the same shape")
			return type(self)([[other.data[j][i] + self.data[j][i] for i in range(self.shape[1])] for j in range(self.shape[0])])
		else:
			raise MatrixException("Matrix can be added only with another matrix")

#!----------------Matrix substracion----------------!#

	def __sub__(self, other):
		if (isinstance(other, Matrix)):
			if ((not isinstance(self, Vector)) and isinstance(other, Vector)):
				return NotImplemented
			if (self.shape != other.shape):
				raise MatrixException("Matrix can be sub only if they have the same shape")
			return type(self)([[self.data[j][i] - other.data[j][i] for i in range(self.shape[1])] for j in range(self.shape[0])])
		else:
			raise MatrixException("Matrix can be sub only with another matrix")

	def __rsub__(self, other):
		if (isinstance(other, Matrix)):
			if ((not isinstance(self, Vector)) and isinstance(other, Vector)):
				return NotImplemented
			if (self.shape != other.shape):
				raise MatrixException("Matrix can be sub only if they have the same shape")
			return type(self)([[other.data[j][i] - self.data[j][i] for i in range(self.shape[1])] for j in range(self.shape[0])])
		else:
			raise MatrixException("Matrix can be sub only with another matrix")

#!----------------Matrix division----------------!#

	def __truediv__(self, scalar):
		if (isinstance(scalar, int) or isinstance(scalar, float)):
			return type(self)([[self.data[j][i] / scalar for i in range(self.shape[1])] for j in range(self.shape[0])])
		else:
			raise MatrixException("Matrix can be divided only by a scalar")

	def __rtruediv__(self, scalar):
		raise MatrixException("Scalar can't be divided by a matrix")

#!----------------Matrix multiplication----------------!#

	def __mul__(self, other):
		if (isinstance(other, Matrix)):
			if ((not isinstance(self, Vector)) and isinstance(other, Vector)):
				return NotImplemented
			if (self.shape[1] != other.shape[0]):
				raise MatrixException("Matrix can be multiplied only if the first matrix have the same number of columns as the second matrix have of rows")
			return type(self)([[sum(self.data[j][i] * other.data[i][k] for i in range(self.shape[1])) for k in range(other.shape[1])] for j in range(self.shape[0])])
		elif (isinstance(other, int) or isinstance(other, float)):
			return type(self)([[self.data[j][i] * other for i in range(self.shape[1])] for j in range(self.shape[0])])
		raise MatrixException("Matrix can be multiplied only with another matrix or a scalar")
	
	def __rmul__(self, other):
		if (isinstance(other, Matrix)):
			if ((not isinstance(self, Vector)) and isinstance(other, Vector)):
				return NotImplemented
			if (self.shape[0] != other.shape[1]):
				raise MatrixException("Matrix can be multiplied only if the first matrix have the same number of columns as the second matrix have of rows")
			return type(self)([[sum(other.data[j][i] * self.data[i][k] for i in range(self.shape[0])) for k in range(self.shape[1])] for j in range(other.shape[0])])
		elif (isinstance(other, int) or isinstance(other, float)):
			return type(self)([[self.data[j][i] * other for i in range(self.shape[1])] for j in range(self.shape[0])])
		raise MatrixException("Matrix can be multiplied only with another matrix or a scalar")

#!----------------Matrix string----------------!#

	def __str__(self):
		string = "Matrix size : " + str(self.shape) + "\nContent : "
		for i in range(self.shape[0]):
			string  += "\n"
			string  += str(self.data[i])
		return string 

	def __repr__(self):
		string  = "Matrix size : " + str(self.shape) + "\nContent : "
		for i in range(self.shape[0]):
			string  += "\n"
			string  += str(self.data[i])
		return str

#!----------------Matrix transposition----------------!#

	def	T(self):
		return type(self)([[self.data[i][j] for i in range(self.shape[0])] for j in range(self.shape[1])])

#!####################################################################################################!#
#!#############################################  VECTOR  #############################################!#
#!####################################################################################################!#

class Vector(Matrix):

#!----------------Vector initialisateur----------------!#

	def __init__(self, matrix_data):
		if (isinstance(matrix_data, list)):
			if not (all(len(it) == len(matrix_data[0]) for it in matrix_data)):
				raise MatrixException("Vector must be rectangular")
			self.data = matrix_data
			self.shape = (len(matrix_data), len(matrix_data[0]))
			if (self.shape[0] != 1 and self.shape[1] != 1):
				raise MatrixException("Vector need to have only one row or one column")
		elif (isinstance(matrix_data, tuple) and len(matrix_data) == 2 and matrix_data[0] > 0 and matrix_data[1] > 0):
			if (matrix_data[0] != 1 and matrix_data[1] != 1):
				raise MatrixException("Vector need to have only one row or one column")
			self.shape = matrix_data
			self.data = [[0 for i in range(self.shape[1])] for j in range(self.shape[0])]
		else:
			raise MatrixException("Vector must be created with a list or a tuple")

#!----------------Vector string----------------!#

	def __str__(self):
		string  = "Vector size : " + str(self.shape) + "\nContent : "
		for i in range(self.shape[0]):
			string  += "\n"
			string  += str(self.data[i])
		return string 

	def __repr__(self):
		str = "Vector size : " + str(self.shape) + "\nContent : "
		for i in range(self.shape[0]):
			str += "\n"
			str += str(self.data[i])
		return str

#!----------------Vector dot_product----------------!#

	def dot(self, other):
		if (isinstance(other, Vector)):
			if (self.shape != other.shape):
				raise MatrixException("Vector can be dot only if they have the same shape")
			if (self.shape[0] == 1):
				return sum(self.data[0][i] * other.data[0][i] for i in range(self.shape[1]))
			else:
				return sum(self.data[i][0] * other.data[i][0] for i in range(self.shape[0]))
		else:
			raise MatrixException("Vector can be dot only with another vector")