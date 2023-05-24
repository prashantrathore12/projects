#FACE RECOGNITION   // FACE RECOGNITION AND ATTENDENCE PROJECT
import cv2
import csv 
import os
import glob
import numpy as np
import face_rec
from datetime import datetime

video_capture = cv2.VideoCapture(0)

#C:\Prashant\vc code proj\pics\dhoni.jpg
#C:\Prashant\vc code proj\pics\kolhi.jpg
#C:\Prashant\vc code proj\pics\rohit.jpg
#C:\Prashant\vc code proj\pics\shami.jpg
dhoni_img = face_rec.load_iamge_file("C:\Prashant\vc code proj\pics\dhoni.jpg")
dhoni_enc = face_rec.face_encoding(dhoni_img)[0]

kolhi_img = face_rec.load_iamge_file("C:\Prashant\vc code proj\pics\kolhi.jpg")
kolhi_enc = face_rec.face_encoding(kolhi_img)[0]

rohit_img = face_rec.load_iamge_file("C:\Prashant\vc code proj\pics\rohit.jpg")
rohit_enc = face_rec.face_encoding(rohit_img)[0]

shami_img = face_rec.load_iamge_file("C:\Prashant\vc code proj\pics\shami.jpg")
shami_enc = face_rec.face_encoding(shami_img)[0]

know_face_encoding = [
    dhoni_enc,
    kolhi_enc,
    rohit_enc,
    shami_enc
]

know_face_names = [
    "dhoni",
    "kolhi",
    "rohit",
    "shami"
]

students = know_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%D")

f = open(current_date+'.csv','w+', newline = '')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame= small_frame[:,:,::-1]
    if s:
        face_locations = face_rec.face_locations(rgb_small_frame)
        face_encodings = face_rec.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_rec.compare_faces(know_face_encoding, face_encoding)
            name = ""
            face_distance = face_rec.face_distance(know_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = know_face_names[best_match_index]

            face_names.append(name)
            if name in know_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("attencence system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()