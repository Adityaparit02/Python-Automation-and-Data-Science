print("-"*40)
print('------------------------------------------------')
print("------------Tickit Pricing Software-------------")
print('------------------------------------------------')
print("-"*40)


print("Please enter your age:")
age=int(input())

if(age<=5):
    print("You have free ENTRY !")

elif(age>5 and age<=18):
    print("Ticket price : 900₹")

elif(age>18 and age<=40):
    print("Ticket price : 1200₹")

else:
    print("Ticket price : 500₹")

    