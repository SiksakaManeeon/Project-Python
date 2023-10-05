from locale import normalize
import time
'''
Reason to Visit
1. Emergency
2. Consulting
3. Take medicine
4. Diagnosis
5. Bathe the wound
class HospitalQueue:
class hospitalueue:
'''    

class HospitalQueue:

    nmQueue = []
    prQueue = []

    def dequeue(self):
        if len(HospitalQueue.prQueue) != 0:
            HospitalQueue.prQueue.pop(0)
        else:
            HospitalQueue.nmQueue.pop(0)
    
    def showQueue(self):
        for i in range(len(HospitalQueue.prQueue)):
            print(HospitalQueue.prQueue[i].name)
            print(HospitalQueue.prQueue[i].reason)
            print(HospitalQueue.prQueue[i].time)
            print()
        for j in range(len(HospitalQueue.nmQueue)):
            print(HospitalQueue.nmQueue[j].name)
            print(HospitalQueue.nmQueue[j].reason)
            print(HospitalQueue.nmQueue[j].time)
            print()
    
class PeopleQueue:

    def __init__(self, name, reason):
        self.name = name
        self.reason = reason
        self.isEmergency = False
        self.time = time.strftime('%H:%M:%S')
        if self.reason == 1:
            self.isEmergency = True
            self.reason = 'Emergency'
            HospitalQueue.prQueue.append(self)
        elif self.reason == 2:
            self.reason = 'Consulting'
            HospitalQueue.nmQueue.append(self)
        elif self.reason == 3:
            self.reason = 'Take medicine'  
            HospitalQueue.nmQueue.append(self)
        elif self.reason == 4:
            self.reason = 'Diagnosis'
            HospitalQueue.nmQueue.append(self)
        elif self.reason == 5:
            self.reason = 'Bathe the wound'
            HospitalQueue.nmQueue.append(self)

    def displayInfo(self):
        print('Name:', self.name, '\nReason:', self.reason, '\nTime:', self.time)

hospital = HospitalQueue()

ppl1 = PeopleQueue('Siksaka Maneeon', 5)
ppl2 = PeopleQueue('Nuey', 2)
ppl3 = PeopleQueue('Kan Nahee', 1)
ppl4 = PeopleQueue('View', 3)
ppl5 = PeopleQueue('Idea', 1)

hospital.dequeue()
hospital.showQueue()

hospital.dequeue()
hospital.showQueue()
