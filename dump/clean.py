f1 = open("in_sent_analysis1","r").read()
f2 = open("in_sent_analysis2","r").read()
f3 = open("in_sent_analysis3","r").read()
f4 = open("in_sent_analysis4","r").read()

ind1 = f1.rfind("\n@@")+1
ind2 = f2.rfind("\n@@")+1
ind3 = f3.rfind("\n@@")+1
ind4 = f4.rfind("\n@@")+1

w1 = open("in_sent_analysis1","w")
w2 = open("in_sent_analysis2","w")
w3 = open("in_sent_analysis3","w")
w4 = open("in_sent_analysis4","w")

w1.write(f1[:ind1])
w2.write(f2[:ind2])
w3.write(f3[:ind3])
w4.write(f4[:ind4])

w1.close()
w2.close()
w3.close()
w4.close()
