
# import sys
# from PyQt5.QtWidgets import QMainWindow,QAction,qApp,QApplication,QPushButton,QHBoxLayout,QVBoxLayout
# from PyQt5.QtGui import  QIcon
#
# class TDUI(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.initUI()
#
#     def initUI(self):
#         self.statusBar().showMessage('Ready')
#
#
#         exitAcction=QAction('exit',self)
#         exitAcction.setShortcut('ctrl+Q')
#         exitAcction.setStatusTip('Exit application')
#         exitAcction.triggered.connect(qApp.quit)
#
#         # importfile = QAction('exit', self)
#         # importfile.setShortcut('ctrl+w')
#         # importfile.setStatusTip('Exit application')
#         # importfile.triggered.connect(qApp.quit)
#
#         self.statusBar()
#         menubar = self.menuBar()
#         fileMenu = menubar.addMenu('&File')
#         fileMenu.addAction(exitAcction)
#
#
#
#         self.setGeometry(800, 500, 800, 500)
#         self.setWindowTitle('人脸三维重建')
#         self.show()
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = TDUI()
#     sys.exit(app.exec_())
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import os
import cv2
from core.interact import interact as io
from core import  pathex
from core import osex
from pathlib import  Path
import ffmpeg
from mtcnn import MTCNN
import glob
import tensorflow as tf

from face_decoder import Face3D
from load_data import *
from preprocess_img import Preprocess

from ui import  Ui_MainWindow


class mctnn_thread(QThread):
    trigger = pyqtSignal()
    _trigger_pic_text=pyqtSignal(str)
    # window=Ui_MainWindow()
    def __init__(self):
        super(mctnn_thread, self).__init__()
        # self.TextEdit=InfoNotifier.g_progress_info
        # self._trigger_pic_text.emit("开始")

    def setname(self,indir="",savepath=""):
        self.indir=indir
        self.savepath=savepath
    def run(self):
        self.FaceReconst()
    def load_graph(self, graph_filename):
        print(graph_filename)
        with tf.gfile.GFile(graph_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        return graph_def
    def MtcnnDectect(self):
        window = Ui_MainWindow()
        # self.NoteTipsEdit.append("\n-------读取人脸信息-------")
        # Ui_MainWindow.setupUi(self.NoteTipsEdit.append("\n-------读取人脸信息-------"))
        window.NoteTipsEdit.append('---------读取人脸信息-------')
        # self._trigger_pic_text.emit('\n-------读取人脸信息-------')
        QApplication.processEvents()
        # self.TextEdit.append("\n-------读取人脸信息-------")


        detector = MTCNN()
        input_dir=self.indir
            # self.Text_Second_lineEdit.text()
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

        img_list = glob.glob(input_dir + '/' + '*.png')
        img_list += glob.glob(input_dir + '/' + '*.jpg')
        cnt=0
        for img_path in img_list:
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            try:
                result = detector.detect_faces(image)
                # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.

                bounding_box = result[0]['box']
                # print(result)

                keypoints = result[0]['keypoints']
                land_mask_file = img_path.replace(".jpg", ".txt")
                file_write = open(land_mask_file, "w")
                x, y = keypoints['left_eye']
                file_write.write(f"{x} {y}\n")
                x, y = keypoints['right_eye']
                file_write.write(f"{x} {y}\n")
                x, y = keypoints['nose']
                file_write.write(f"{x} {y}\n")
                x, y = keypoints['mouth_left']
                file_write.write(f"{x} {y}\n")
                x, y = keypoints['mouth_right']
                file_write.write(f"{x} {y}\n")
            except:
                cnt += 1
        # self.NoteTipsEdit.append("\n-------有{}张提取失败-------".format(cnt))
        self._trigger_pic_text.emit("\n-------有{}张提取失败-------".format(cnt))
        QApplication.processEvents()
        # self.TextEdit.append("\n-------有{}张提取失败-------".format(cnt))


    def FaceReconst(self):
        self.MtcnnDectect()

        inFileDir=self.indir
        save_path=self.savepath
        img_list=glob.glob(inFileDir+'/'+'*.png')
        img_list+=glob.glob(inFileDir+'/'+'*.jpg')
        # read BFM face model
        # transfer original BFM model to our model
        if not os.path.isfile('./BFM/BFM_model_front.mat'):
            transferBFM09()

        # read standard landmarks for preprocessing images
        lm3D = load_lm3d()
        batchsize = 1
        n = 0

        with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
            FaceReconstructor = Face3D()
            images = tf.placeholder(name='input_imgs', shape=[batchsize, 224, 224, 3], dtype=tf.float32)
            graph_def = self.load_graph('network/FaceReconModel.pb')
            tf.import_graph_def(graph_def, name='resnet', input_map={'input_imgs:0': images})

            # output coefficients of R-Net (dim = 257)
            coeff = graph.get_tensor_by_name('resnet/coeff:0')

            # reconstructing faces
            FaceReconstructor.Reconstruction_Block(coeff, batchsize)
            face_shape = FaceReconstructor.face_shape_t
            face_texture = FaceReconstructor.face_texture
            face_color = FaceReconstructor.face_color
            landmarks_2d = FaceReconstructor.landmark_p
            # recon_img = FaceReconstructor.render_imgs
            tri = FaceReconstructor.facemodel.face_buf
            with tf.Session() as sess:
                # print('-----------reconstructing-----------')
                # self.TextEdit.append("\n-------正在重建-------")
                # self.NoteTipsEdit.append("\n-------正在重建-------")
                self._trigger_pic_text.emit("\n-------正在重建-------")
                QApplication.processEvents()
                cnt=0
                for file in img_list:
                    try:
                        n+=1
                        print(n)
                        # self.NoteTipsEdit.append("\n第"+n+"张正在重建")
                        # load images and corresponding 5 facial landmarks
                        img, lm = load_img(file, file.replace('png', 'txt').replace('jpg', 'txt'))
                        # preprocess input image
                        input_img, lm_new, transform_params = Preprocess(img, lm, lm3D)

                        # coeff_,face_shape_,face_texture_,face_color_,landmarks_2d_,recon_img_,tri_ = sess.run([coeff,\
                        # 	face_shape,face_texture,face_color,landmarks_2d,recon_img,tri],feed_dict = {images: input_img})

                        coeff_, face_shape_, face_texture_, face_color_, landmarks_2d_, tri_ = sess.run([coeff, face_shape,
                                                                                                         face_texture,
                                                                                                         face_color,
                                                                                                         landmarks_2d, tri],
                                                                                                        feed_dict={
                                                                                                            images: input_img})

                        # reshape outputs
                        input_img = np.squeeze(input_img)
                        face_shape_ = np.squeeze(face_shape_, (0))
                        face_texture_ = np.squeeze(face_texture_, (0))
                        face_color_ = np.squeeze(face_color_, (0))
                        landmarks_2d_ = np.squeeze(landmarks_2d_, (0))
                        # recon_img_ = np.squeeze(recon_img_, (0))

                        # save output files
                        # savemat(os.path.join(save_path,file.split(os.path.sep)[-1].replace('.png','.mat').replace('jpg','mat')),{'cropped_img':input_img[:,:,::-1],'recon_img':recon_img_,'coeff':coeff_,\
                        # 	'face_shape':face_shape_,'face_texture':face_texture_,'face_color':face_color_,'lm_68p':landmarks_2d_,'lm_5p':lm_new})
                        # savemat(os.path.join(save_path,file.split(os.path.sep)[-1].replace('.png','.mat').replace('jpg','mat')),{'cropped_img':input_img[:,:,::-1],'coeff':coeff_,\
                        # 	'face_shape':face_shape_,'face_texture':face_texture_,'face_color':face_color_,'lm_68p':landmarks_2d_,'lm_5p':lm_new})
                        save_obj(
                            os.path.join(save_path, file.split(os.path.sep)[-1].replace('.png', '_mesh.obj').replace('jpg',
                                                                                                                     '_mesh.obj')),
                            face_shape_, tri_,
                            np.clip(face_color_, 0, 255) / 255)  # 3D reconstruction face (in canonical view)
                    except:
                        cnt += 1
                # self.NoteTipsEdit.append("\n-------有{}张重建失败-------".format(cnt))
                # self.NoteTipsEdit.append("\n-------完毕-------")
                self._trigger_pic_text.emit("\n-------有{}张重建失败-------".format(cnt))
                self._trigger_pic_text.emit("\n-------完毕-------")
                # self.TextEdit.append("\n-------有{}张重建失败-------".format(cnt))
                # self.TextEdit.append("\n-------完毕-------")
                QApplication.processEvents()



        self.trigger.emit()



class mctnn_SpePic_thread(QThread):
    _trigger=pyqtSignal()
    _trigger_text=pyqtSignal(str)
    def __init__(self):
        super(mctnn_SpePic_thread,self).__init__()
        # self.Tipstext=InfoNotifier.g_progress_info
    def setname(self,indir=[],savepath=""):
        self.indir=indir
        self.savepath=savepath
    def load_graph(self, graph_filename):
        print(graph_filename)
        with tf.gfile.GFile(graph_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        return graph_def
    def mtcnn_Spec_pic(self):
        inDir = self.indir
        # Text_Forth_lineEdit.text().split(';;')
        # save_path = self.savepath
        # Text_Fifth_lineEdit.text()
        # self.NoteTipsEdit.append("\n-------读取人脸信息-------")
        # self.Tipstext.append("\n-------读取人脸信息-------")
        self._trigger_text.emit('\n-------读取人脸信息-------')
        QApplication.processEvents()
        detector = MTCNN()
        cnt = 0
        for img_path in inDir:
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            try:
                result = detector.detect_faces(image)
                # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.

                bounding_box = result[0]['box']
                # print(result)

                keypoints = result[0]['keypoints']
                land_mask_file = img_path.replace(".jpg", ".txt")
                file_write = open(land_mask_file, "w")
                x, y = keypoints['left_eye']
                file_write.write(f"{x} {y}\n")
                x, y = keypoints['right_eye']
                file_write.write(f"{x} {y}\n")
                x, y = keypoints['nose']
                file_write.write(f"{x} {y}\n")
                x, y = keypoints['mouth_left']
                file_write.write(f"{x} {y}\n")
                x, y = keypoints['mouth_right']
                file_write.write(f"{x} {y}\n")
                file_write.close()
            except:
                cnt += 1
        self._trigger_text.emit("\n-------有{}张提取失败-------".format(cnt))
        # self.Tipstext.append("\n-------有{}张提取失败-------".format(cnt))
        QApplication.processEvents()

        # self.NoteTipsEdit.append("\n-------有{}张提取失败-------".format(cnt))
    def FaceReconst_Spec_pics(self):
        inDir = self.indir
            # Text_Forth_lineEdit.text().split(';;')
        save_path = self.savepath
        self.mtcnn_Spec_pic()
            # Text_Fifth_lineEdit.text()
        # self.NoteTipsEdit.append("\n-------读取人脸信息-------")
        # detector = MTCNN()
        # cnt = 0
        # for img_path in inDir:
        #     image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        #     try:
        #         result = detector.detect_faces(image)
        #         # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
        #
        #         bounding_box = result[0]['box']
        #         # print(result)
        #
        #         keypoints = result[0]['keypoints']
        #         land_mask_file = img_path.replace(".jpg", ".txt")
        #         file_write = open(land_mask_file, "w")
        #         x, y = keypoints['left_eye']
        #         file_write.write(f"{x} {y}\n")
        #         x, y = keypoints['right_eye']
        #         file_write.write(f"{x} {y}\n")
        #         x, y = keypoints['nose']
        #         file_write.write(f"{x} {y}\n")
        #         x, y = keypoints['mouth_left']
        #         file_write.write(f"{x} {y}\n")
        #         x, y = keypoints['mouth_right']
        #         file_write.write(f"{x} {y}\n")
        #         file_write.close()
        #     except:
        #         cnt += 1
        # self._trigger_text.emit("\n-------有{}张提取失败-------".format(cnt))
        # # self.NoteTipsEdit.append("\n-------有{}张提取失败-------".format(cnt))
        import  time
        # time.sleep(1)
        if not os.path.isfile('./BFM/BFM_model_front.mat'):
            transferBFM09()

        # read standard landmarks for preprocessing images
        lm3D = load_lm3d()
        batchsize = 1
        n = 0

        with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
            FaceReconstructor = Face3D()
            images = tf.placeholder(name='input_imgs', shape=[batchsize, 224, 224, 3], dtype=tf.float32)
            graph_def = self.load_graph('network/FaceReconModel.pb')
            tf.import_graph_def(graph_def, name='resnet', input_map={'input_imgs:0': images})

            # output coefficients of R-Net (dim = 257)
            coeff = graph.get_tensor_by_name('resnet/coeff:0')

            # reconstructing faces
            FaceReconstructor.Reconstruction_Block(coeff, batchsize)
            face_shape = FaceReconstructor.face_shape_t
            face_texture = FaceReconstructor.face_texture
            face_color = FaceReconstructor.face_color
            landmarks_2d = FaceReconstructor.landmark_p
            # recon_img = FaceReconstructor.render_imgs
            tri = FaceReconstructor.facemodel.face_buf
            with tf.Session() as sess:
                # print('-----------reconstructing-----------')
                # self.NoteTipsEdit.append("\n-------正在重建-------")
                self._trigger_text.emit("\n-------正在重建-------")
                QApplication.processEvents()
                # self.Tipstext.append("\n-------正在重建-------")
                cntt=0

                for file in inDir:
                    try:
                        n+=1
                        print(n)
                        # print(file)
                        # self.NoteTipsEdit.append("\n第"+n+"张正在重建")
                        # load images and corresponding 5 facial landmarks
                        filetxt=os.path.basename(file)
                        # print(filetxt)
                        img, lm = load_img(file, file.replace('png', 'txt').replace('jpg', 'txt'))
                        # print(lm)
                        # preprocess input image
                        input_img, lm_new, transform_params = Preprocess(img, lm, lm3D)

                        # coeff_,face_shape_,face_texture_,face_color_,landmarks_2d_,recon_img_,tri_ = sess.run([coeff,\
                        # 	face_shape,face_texture,face_color,landmarks_2d,recon_img,tri],feed_dict = {images: input_img})

                        coeff_, face_shape_, face_texture_, face_color_, landmarks_2d_, tri_ = sess.run([coeff, face_shape,
                                                                                                         face_texture,
                                                                                                         face_color,
                                                                                                         landmarks_2d, tri],
                                                                                                        feed_dict={
                                                                                                            images: input_img})

                        # reshape outputs
                        input_img = np.squeeze(input_img)
                        face_shape_ = np.squeeze(face_shape_, (0))
                        face_texture_ = np.squeeze(face_texture_, (0))
                        face_color_ = np.squeeze(face_color_, (0))
                        landmarks_2d_ = np.squeeze(landmarks_2d_, (0))
                        # recon_img_ = np.squeeze(recon_img_, (0))

                        # save output files
                        # savemat(os.path.join(save_path,file.split(os.path.sep)[-1].replace('.png','.mat').replace('jpg','mat')),{'cropped_img':input_img[:,:,::-1],'recon_img':recon_img_,'coeff':coeff_,\
                        # 	'face_shape':face_shape_,'face_texture':face_texture_,'face_color':face_color_,'lm_68p':landmarks_2d_,'lm_5p':lm_new})
                        # savemat(os.path.join(save_path,file.split(os.path.sep)[-1].replace('.png','.mat').replace('jpg','mat')),{'cropped_img':input_img[:,:,::-1],'coeff':coeff_,\
                        # 	'face_shape':face_shape_,'face_texture':face_texture_,'face_color':face_color_,'lm_68p':landmarks_2d_,'lm_5p':lm_new})
                        save_obj(
                            os.path.join(save_path, filetxt.replace('.png', '_mesh.obj').replace('.jpg','_mesh.obj')),
                            face_shape_, tri_,
                            np.clip(face_color_, 0, 255) / 255)  # 3D reconstruction face (in canonical view)
                    except:
                        cntt +=1
                # self.NoteTipsEdit.append("\n-------有{}张重建失败-------".format(cnt))
                # self.NoteTipsEdit.append("\n-------完毕-------")
                print(cntt)
                self._trigger_text.emit("\n-------有{}张重建失败-------".format(cntt))
                self._trigger_text.emit("\n-------完毕-------")
                QApplication.processEvents()
                # self.Tipstext.append("\n-------有{}张重建失败-------".format(cntt))
                # self.Tipstext.append("\n-------完毕-------")


        self._trigger.emit()
    def run(self):
        self.FaceReconst_Spec_pics()
