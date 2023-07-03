#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2


# In[4]:


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[5]:


cap = cv2.VideoCapture(0)


# In[ ]:


while True:
 
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        blurred_roi = cv2.blur(face_roi, (50, 50))
        frame[y:y + h, x:x + w] = blurred_roi
    cv2.imshow('Face Blur', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# In[ ]:


cap.release()


# In[ ]:


cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




