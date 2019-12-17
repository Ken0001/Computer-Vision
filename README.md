# Computer-Vision
## Practice 10 (Homework 3)
1. Input BGR images from webcam.
2. Use cv2.CascadeClassifier() to detect your face, mouth, and eyes.
3. Input BGRA images from files "mustache.png" and "hat.png"
4. Perform <b> Alpha Blending </b> to add mustache and hat on the right position and orientation of your face.
5. The overlaid mustache and hat should be translated, rotated, and scaled according the movement of your face. 
6. Show your output images.
7. Upload your Jupyter code file (*.ipynb)

## 說明
去背
---
```python
def addImg(roi, img2):
```
這裡是取得帽子跟鬍子的去背圖
`roi`是背景, `img2`是要貼的物件

Detection
---
```python
cv2.CascadeClassifier().detectMultiScale()
```
`CascadeClassifier()`裡面放xml
`detectMultiScale()`的參數會影響detection結果, 可以研究一下

Face
---
```python
sw = (w/304)
sh = int(277*sw)
hat = cv2.resize(hat, (w, sh), interpolation=cv2.INTER_CUBIC)
roi = frame[y-int(0.6*h):y+sh-int(0.6*h), x:x+w]
#print("ROI",roi.shape,",HAT",mus.shape," ALL",(x,y,w,h))
dst = getMask(roi, hat)
frame[y-int(0.6*h):y+sh-int(0.6*h), x:x+w] = dst)
```
`sw`, `sh`依照臉寬等比例縮小帽子
物件蓋上的方法是先抓一塊要放物件的區塊(注意size要跟物件size一樣)
不過物件不知道為啥放進來都有白色背景, 所以我手動去背(`getMas()`)

Mouth
---
```python
nx = x-int(0.5*w)
nX = x+int(0.5*w)+w
ny = y-int(0.5*h)
nY = y-int(0.5*h)+int(1.35*h)
```
跟上面一樣

Eyes
---
呃 範例有但作業沒用到???

注意
---
感謝你的注意<br>
第3.中提到要用BGRA代表讀圖要用`hat = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)`<br>
我猜他可能想用`cv2.addWeight()`方式做這作業<br>
```python
# cv2.addWeighted(背景, 背景透明度比例, 物件, 物件透明度比例, Gamma值)
add = cv2.addWeighted(frame[y:y+h, x:x+w, :], alpha, hat, 1-alpha, 0)
# 把一樣的位置取代掉
frame[y:y+h, x:x+w, :] = add 
```
這不難但會有白色背景
註解裡面有
