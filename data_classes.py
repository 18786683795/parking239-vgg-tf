import numpy as np

classes = ["free", "occupied", "obscure", "closed"]
_one_hot = np.eye(len(classes))


class_one_hot_by_name = {}

for i in range(len(classes)):
	class_one_hot_by_name[classes[i]] = _one_hot[i]