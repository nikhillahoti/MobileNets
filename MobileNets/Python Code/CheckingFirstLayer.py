

counter = 0
fMy = open("data/LayerOne/output.txt", "r")
myContent = fMy.readlines()
fAc = open("First_Layer_Output.txt", "r")
acContent = fAc.readlines()

iterr = -1
iterrMy = -1
position = []
mismatchValues = []
countZeros = 0
for i in range(0, 32):
    for j in range(0, 114):
        for k in range(0, 114):

            iterrMy += 1

            if k > 111 or j > 111:
                if float(myContent[iterrMy]).is_integer(): countZeros += 1
                continue

            iterr += 1

            if int(round(float(myContent[iterrMy].strip()))) == int(round(float(acContent[iterr].strip()))):
                counter += 1
            else:
                position.append((i,j,k))
                mismatchValues.append((myContent[iterrMy].strip(), acContent[iterr].strip()))
fMy.close()
fAc.close()
print("Total Match ---> ", counter)
print("Total Mismatch ---> " + str(len(position)))
print(iterrMy)
print(countZeros)

#for i in range(len(position)):
#    print(position[i])
if len(position) > 10:
    for i in range(len(position)):
        print(position[i])
        print(mismatchValues[i])

