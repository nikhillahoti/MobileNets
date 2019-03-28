

fOrg = open('output-Org.txt')
fOrgLines = fOrg.readlines()


fGen = open('output.txt')
fGenLines = fGen.readlines()

count = 0
for i in range(len(fGenLines)):
	if(fGenLines[i] != fOrgLines[i]):
		count += 1
		if count < 30:
			print(fGenLines[i] + "----" + fOrgLines[i])

print("Number of mismatches: " + str(count))
