import pdb

def stack_to_string(stack):
	op = ""
	for i in stack:
		if op == "":
			op = op + i
		else:
			op = op + ' ' + i
	return op

def cal_score(outputs, trgs):
	corr = 0
	tot = 0
	disp_corr = []
	for i in range(len(outputs)):
		op = stack_to_string(outputs[i])
		if op == trgs[i]:
			corr+=1
			tot+=1
			disp_corr.append(1)
		else:
			tot+=1
			disp_corr.append(0)

	return corr, tot, disp_corr
