

counter = 0
fMy = open("output.txt", "r")
myContent = fMy.readlines()
fAc = open("First_Layer_Output.txt", "r")
acContent = fAc.readlines()

iterr = -1
position = []
for i in range(0, 32):
    for j in range(0, 112):
        for k in range(0, 112):
            iterr += 1
            #if k > 95: continue

            if int(float(myContent[iterr].strip())) == int(float(acContent[iterr].strip())):
                counter += 1
            else:
                position.append((i,j,k))
                print(myContent[iterr].strip())
                print(acContent[iterr].strip())
                print("----")
fMy.close()
fAc.close()
print("Total Match ", counter)
print(iterr)
print(len(position))

for i in range(len(position)):
    print(position[i])
#for i in range(10):
#    print(position[i])

