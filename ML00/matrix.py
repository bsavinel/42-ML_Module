class Matrix:
	def __init__(self, matrix_data):
		if isinstance(matrix_data, list):
			if not all(len(it) == len(matrix_data[0]) for it in matrix_data):
				raise ValueError("Matrix must be rectangular")
			self.data = matrix_data
			self.shape = (len(matrix_data), len(matrix_data[0]))
		else:
			self.shape = matrix_data
			self.data = [[0 for i in range(self.shape[1])] for j in range(self.shape[0])]
	
	def __add__(self, other):
		if (isinstance(other, Matrix)):
			if (self.shape != other.shape):
				print("Matrix can be added only if they have the same shape")
				return NotImplemented
			tab = [[self.data[j][i] + other.data[j][i] for i in range(self.shape[1])] for j in range(self.shape[0])]
		else:
			return NotImplemented

	def __radd__(self, other):
		if (isinstance(other, Matrix)):
			if (self.shape != other.shape):
				print("Matrix can be added only if they have the same shape")
				return NotImplemented
			tab = [[other.data[j][i] + self.data[j][i] for i in range(self.shape[1])] for j in range(self.shape[0])]
		else:
			return NotImplemented

	def __sub__(self, other):
		if (isinstance(other, Matrix)):
			if (self.shape != other.shape):
				print("Matrix can be sub only if they have the same shape")
				return NotImplemented
			tab = [[self.data[j][i] - other.data[j][i] for i in range(self.shape[1])] for j in range(self.shape[0])]
		else:
			return NotImplemented

	def __rsub__(self, other):
		if (isinstance(other, Matrix)):
			if (self.shape != other.shape):
				print("Matrix can be sub only if they have the same shape")
				return NotImplemented
			tab = [[other.data[j][i] - self.data[j][i] for i in range(self.shape[1])] for j in range(self.shape[0])]
		else:
			return NotImplemented
		
	
	
