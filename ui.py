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
import InfoNotifier
from face_decoder import Face3D
from load_data import *
from preprocess_img import Preprocess

import time
class Mythread(QThread):
    _signal_progress_info = pyqtSignal()

    _signal_button_ctrl = pyqtSignal()

    def __init__(self):
        super(Mythread, self).__init__()

    def run(self):
        while True:
            # 发出信号
            self._signal_progress_info.emit()
            self._signal_button_ctrl.emit()
            # 让程序休眠
            time.sleep(1.5)

class Ui_MainWindow(QtWidgets.QMainWindow):
    mysignal=pyqtSignal(str)
    def __init__(self):
        super(Ui_MainWindow,self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

        self.cwd = os.getcwd()
        self.ffmpeg_cmd_path = self.cwd + "\\ffmpeg\\ffmpeg".replace('/','\\')
        self.ffprobe_cmd_path = self.cwd + "\\ffmpeg\\ffprobe".replace('/','\\')
        # print(self.ffmpeg_cmd_path)
        self.g_loss_info=""
        self.g_progress_info = []
        # self.statusBar().showMessage('Ready')
        self.thread1 = mctnn_thread()
        self.thread2 = mctnn_SpePic_thread()
        # self.mysignal.connect(self.)
        # self.thread1._trigger_pic_text.connect(self.update_text)
        # self.thread2._trigger_text.connect(self.update_text)
        self.Thread = Mythread()
        self.Thread._signal_progress_info.connect(self.update_progress_info)
        self.Thread.start()






    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 600)
        # MainWindow.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        MainWindow.setFixedSize(MainWindow.width(), MainWindow.height())
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.retranslateUi(MainWindow)
        MainWindow.setCentralWidget(self.centralWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        self.Text_first = QtWidgets.QLabel(self.centralWidget)
        self.Text_first.setGeometry(QtCore.QRect(60, 90, 120, 23))
        self.Text_first.setText("第一步：选择目标视频文件")
        self.Text_first.setObjectName("Text_first")
        MainWindow.setCentralWidget(self.centralWidget)

        self.ChooseVideoBTN = QtWidgets.QPushButton(self.centralWidget)
        self.ChooseVideoBTN.setGeometry(QtCore.QRect(600, 90, 120, 23))
        self.ChooseVideoBTN.setObjectName("pushButton")
        self.ChooseVideoBTN.setText("选择视频")
        MainWindow.setCentralWidget(self.centralWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.Text_first_lineEdit = QtWidgets.QLineEdit(self.centralWidget)
        self.Text_first_lineEdit.setGeometry(QtCore.QRect(200, 90, 350, 23))
        self.Text_first_lineEdit.setObjectName("Text_first_lineEdit")

        self.Text_second = QtWidgets.QLabel(self.centralWidget)
        self.Text_second.setGeometry(QtCore.QRect(60, 190, 120, 23))
        self.Text_second.setText("第二步：从视频中提取人脸图像")
        self.Text_second.setObjectName("Text_second")
        MainWindow.setCentralWidget(self.centralWidget)

        self.Text_fps = QtWidgets.QLabel(self.centralWidget)
        self.Text_fps.setGeometry(QtCore.QRect(450, 220, 60, 23))
        self.Text_fps.setText("设置码率：")
        self.Text_fps.setObjectName("Text_fps")
        MainWindow.setCentralWidget(self.centralWidget)

        self.setFpsEdit = QtWidgets.QLineEdit(self.centralWidget)
        self.setFpsEdit.setGeometry(QtCore.QRect(520, 220, 50, 23))
        self.setFpsEdit.setObjectName("setFpsEdit")
        self.setFpsEdit.setText("Default")
        self.setFpsEdit.setToolTip("Default=24")

        self.ChooseDir_frame = QtWidgets.QPushButton(self.centralWidget)
        self.ChooseDir_frame.setGeometry(QtCore.QRect(600, 190, 120, 23))
        self.ChooseDir_frame.setObjectName("选择提取的人脸图像输出路径")
        self.ChooseDir_frame.setText("选择目录")
        MainWindow.setCentralWidget(self.centralWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        self.GetFrame = QtWidgets.QPushButton(self.centralWidget)
        self.GetFrame.setGeometry(QtCore.QRect(600, 220, 120, 23))
        self.GetFrame.setObjectName("GetFrame")
        self.GetFrame.setText("开始提取图像")
        MainWindow.setCentralWidget(self.centralWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.Text_Second_lineEdit = QtWidgets.QLineEdit(self.centralWidget)
        self.Text_Second_lineEdit.setGeometry(QtCore.QRect(200, 190, 350, 23))
        self.Text_Second_lineEdit.setObjectName("Text_Second_lineEdit")

        self.Text_third = QtWidgets.QLabel(self.centralWidget)
        self.Text_third.setGeometry(QtCore.QRect(60, 290, 120, 23))
        self.Text_third.setText("第三步：开始三维重建")
        self.Text_third.setObjectName("Text_third")
        MainWindow.setCentralWidget(self.centralWidget)

        self.Text_Third_lineEdit = QtWidgets.QLineEdit(self.centralWidget)
        self.Text_Third_lineEdit.setGeometry(QtCore.QRect(200, 290, 350, 23))
        self.Text_Third_lineEdit.setObjectName("Text_Third_lineEdit")

        self.NoteTips = QtWidgets.QLabel(self.centralWidget)
        self.NoteTips.setGeometry(QtCore.QRect(750, 90, 120, 23))
        # textInTips=InfoNotifier.g_progress_info
        self.NoteTips.setText("进度")
        self.NoteTips.setObjectName("NoteTips")
        MainWindow.setCentralWidget(self.centralWidget)
        # self.NoteTipsEdit = QtWidgets.QTextEdit(self.centralWidget)
        self.NoteTipsEdit = QtWidgets.QTextEdit(self.centralWidget)
        self.NoteTipsEdit.setGeometry(QtCore.QRect(750, 120, 200, 400))
        self.NoteTipsEdit.setObjectName("进度窗口")
        # self.NoteTipsEdit.



        self.SelectOutDir = QtWidgets.QPushButton(self.centralWidget)
        self.SelectOutDir.setGeometry(QtCore.QRect(600, 290, 120, 23))
        self.SelectOutDir.setObjectName("SelectOutDir")
        self.SelectOutDir.setText("选择输出路径")
        MainWindow.setCentralWidget(self.centralWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.FrameRestruct = QtWidgets.QPushButton(self.centralWidget)
        self.FrameRestruct.setGeometry(QtCore.QRect(600, 320, 120, 23))
        self.FrameRestruct.setObjectName("FrameRestruct")
        self.FrameRestruct.setText("开始重建人脸")
        # self.FrameRestruct.setToolTip("重建前请把文件夹中不含全脸的帧删除！")
        MainWindow.setCentralWidget(self.centralWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.split_Line = QtWidgets.QLabel(self.centralWidget)
        self.split_Line.setGeometry(QtCore.QRect(60, 350, 700, 5))
        self.split_Line.setText("---------------------------------------------------------------------------------------")
        self.split_Line.setObjectName("split_Line")
        MainWindow.setCentralWidget(self.centralWidget)

        self.Single_pic_text = QtWidgets.QLabel(self.centralWidget)
        self.Single_pic_text.setGeometry(QtCore.QRect(50, 355, 120, 25))
        self.Single_pic_text.setText(
            "特定图片操作")
        self.Single_pic_text.setObjectName("Single_pic_text")
        MainWindow.setCentralWidget(self.centralWidget)

        self.Choose_pic_dir = QtWidgets.QLabel(self.centralWidget)
        self.Choose_pic_dir.setGeometry(QtCore.QRect(90, 390, 120, 25))
        self.Choose_pic_dir.setText(
            "选取图片:")
        self.Single_pic_text.setObjectName("Choose_pic_dir")
        MainWindow.setCentralWidget(self.centralWidget)

        self.Text_Forth_lineEdit = QtWidgets.QLineEdit(self.centralWidget)
        self.Text_Forth_lineEdit.setGeometry(QtCore.QRect(200, 390, 350, 23))
        self.Text_Forth_lineEdit.setObjectName("Text_Forth_lineEdit")

        self.Select_pic_dir = QtWidgets.QPushButton(self.centralWidget)
        self.Select_pic_dir.setGeometry(QtCore.QRect(600, 390, 120, 23))
        self.Select_pic_dir.setObjectName("Select_pic_dir")
        self.Select_pic_dir.setText("选择图片")
        MainWindow.setCentralWidget(self.centralWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.Rec_Pic_Text = QtWidgets.QLabel(self.centralWidget)
        self.Rec_Pic_Text.setGeometry(QtCore.QRect(90, 490, 120, 25))
        self.Rec_Pic_Text.setText(
            "开始重建:")
        self.Single_pic_text.setObjectName("Rec_Pic_Text")
        MainWindow.setCentralWidget(self.centralWidget)

        self.Text_Fifth_lineEdit = QtWidgets.QLineEdit(self.centralWidget)
        self.Text_Fifth_lineEdit.setGeometry(QtCore.QRect(200, 490, 350, 23))
        self.Text_Fifth_lineEdit.setObjectName("Text_Fifth_lineEdit")

        self.Select_pic_out_dir = QtWidgets.QPushButton(self.centralWidget)
        self.Select_pic_out_dir.setGeometry(QtCore.QRect(600, 490, 120, 23))
        self.Select_pic_out_dir.setObjectName("Select_pic_out_dir")
        self.Select_pic_out_dir.setText("选择输出路径")
        MainWindow.setCentralWidget(self.centralWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.PicRestruct = QtWidgets.QPushButton(self.centralWidget)
        self.PicRestruct.setGeometry(QtCore.QRect(600, 520, 120, 23))
        self.PicRestruct.setObjectName("PicRestruct")
        self.PicRestruct.setText("开始重建人脸")
        # self.FrameRestruct.setToolTip("重建前请把文件夹中不含全脸的帧删除！")
        MainWindow.setCentralWidget(self.centralWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)








        self.ChooseVideoBTN.clicked.connect(self.openfile)
        self.SelectOutDir.clicked.connect(self.getOutDir_Fianl)
        self.ChooseDir_frame.clicked.connect(self.getOutDir_Frame)
        self.GetFrame.clicked.connect(self.extract_dst_video)
        self.FrameRestruct.clicked.connect(self.FaceReconst_clicked)
        self.Select_pic_dir.clicked.connect(self.getoutDir_SinglePic)
        self.Select_pic_out_dir.clicked.connect(self.save_outDir_single)
        self.PicRestruct.clicked.connect(self.FaceReconst_SpecPic_clicked)

    def FaceReconst_clicked(self):
        self.thread_faceconst=mctnn_thread()
        indir=self.Text_Second_lineEdit.text()
        savepath=self.Text_Third_lineEdit.text()
        self.thread_faceconst.setname(indir, savepath)
        # self.thread_faceconst.trigger.connect(self.callback)
        self.thread_faceconst.start()
    def FaceReconst_SpecPic_clicked(self):
        self.thread_specPic_const=mctnn_SpePic_thread()
        indir=self.Text_Forth_lineEdit.text().split(';;')
        savepath=self.Text_Fifth_lineEdit.text()
        self.thread_specPic_const.setname(indir,savepath)
        self.thread_specPic_const.run()
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "三维人脸重建-剑网三"))

    def openfile(self):
        filename, filetype = QFileDialog.getOpenFileName(self, '选择文件', '', 'MP4 Files (*.mp4);;AVI Files (*.avi);;FLV Files (*.flv)')
        self.Text_first_lineEdit.setText(filename)
        # self.NoteTipsEdit.setPlainText("视频路径："+filename)
        # cap=cv2.VideoCapture(filename)
        # self.NoteTipsEdit.setPlainText(cap.get(5))


        # print(fileName1, filetype)
    def update_progress_info(self):
        # self.NoteTipsEdit.insertPlainText(message)
        for info in InfoNotifier.InfoNotifier.g_progress_info:
            self.NoteTipsEdit.append(info)
        InfoNotifier.InfoNotifier.g_progress_info.clear()
    def getOutDir_Fianl(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "获取文件路径", "./")
        # 当窗口非继承QtWidgets.QDialog时，self可替换成 None
        self.Text_Third_lineEdit.setText(directory)
        # self.NoteTipsEdit.append("输出路径：" + directory)
    def getOutDir_Frame(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "获取文件路径", "./")
        self.Text_Second_lineEdit.setText(directory)
    def getoutDir_SinglePic(self):
        files,filetype=QFileDialog.getOpenFileNames(self,"选择文件","./","JPG文件(*.jpg);;PNG文件(*.png)")
        if len(files) == 0:
            return
        pic_dir=""
        for file in files:
            pic_dir+=file+';;'
        pic_dir=pic_dir[:-2]
        self.Text_Forth_lineEdit.setText(pic_dir)
    def save_outDir_single(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "获取文件路径", "./")
        self.Text_Fifth_lineEdit.setText(directory)

    def extract_video(self,input_file, output_dir, output_ext=None, fps=None):
        input_file_path = Path(input_file)
        output_path = Path(output_dir)

        if not output_path.exists():
            output_path.mkdir(exist_ok=True)
        # self.g_progress_info.append("\n视频帧输出目录: " + str(Path(output_path).absolute()))
        # self.NoteTipsEdit.append("\n视频帧输出目录: " + str(Path(output_path).absolute()))
        InfoNotifier.InfoNotifier.g_progress_info.append("\n视频帧输出目录: " + str(Path(output_path).absolute()))


        if input_file_path.suffix == '.*':
            input_file_path = pathex.get_first_file_by_stem(input_file_path.parent, input_file_path.stem)
        else:
            if not input_file_path.exists():
                input_file_path = None

        # self.g_progress_info.append("\n视频输入路径:" + str(input_file_path))
        # self.NoteTipsEdit.append("\n视频输入路径:" + str(input_file_path))
        InfoNotifier.InfoNotifier.g_progress_info.append("\n视频输入路径:" + str(input_file_path))

        if input_file_path is None:
            io.log_err("input_file not found.")
            # self.g_progress_info.append("\n视频输入路径不存在")
            # self.NoteTipsEdit.append("\n视频输入路径不存在")
            InfoNotifier.InfoNotifier.g_progress_info.append("\n视频输入路径不存在")
            return

        # if fps is None:
        #     fps = io.input_int("Enter FPS", 0,
        #                        help_message="How many frames of every second of the video will be extracted. 0 - full fps")
            # self.NoteTipsEdit.append("\n读取不到帧")

        # self.g_progress_info.append("\n视频帧抽取频率: full fps")
        # self.NoteTipsEdit.append("\n视频帧抽取频率: full fps")

        if output_ext is None:
            output_ext = io.input_str("Output image format", "png", ["png", "jpg"],
                                      help_message="png is lossless, but extraction is x10 slower for HDD, requires x10 more disk space than jpg.")

        # self.g_progress_info.append("\n视频帧输出格式频率: " + output_ext)
        # self.NoteTipsEdit.append("\n视频帧输出格式频率: " + output_ext)
        InfoNotifier.InfoNotifier.g_progress_info.append("\n视频帧输出格式频率: " + output_ext)

        filenames = pathex.get_image_paths(output_path, ['.' + output_ext])
        if len(filenames) != 0:
            # self.g_progress_info.append("\n视频帧输出目录不为空, 该目录将被清空!")
            InfoNotifier.InfoNotifier.g_progress_info.append("\n视频帧输出目录不为空, 该目录将被清空!")
            # self.NoteTipsEdit.append("\n视频帧输出目录不为空, 该目录将被清空!")
            # Ui_MainWindow.setupUi(self.NoteTipsEdit.append())

        for filename in filenames:
            Path(filename).unlink()
            QApplication.processEvents()

        job = ffmpeg.input(str(input_file_path))

        kwargs = {'pix_fmt': 'rgb24'}
        # if fps !=0:
        kwargs.update({'r': str(fps)})

        if output_ext == 'jpg':
            kwargs.update({'q:v': '2'})  # highest quality for jpg

        job = job.output(str(output_path / ('%5d.' + output_ext)), **kwargs)

        try:
            job, err = job.run(cmd=self.ffmpeg_cmd_path)
        except:
            io.log_err("ffmpeg fail, job commandline:" + str(job.compile()))

        # cmd = 'E:\\Users\\shishaohua.SHISHAOHUA1\\Downloads\\DeepFaceLab_NVIDIA\\_internal\\ffmpeg\\ffmpeg -i E:\\Users\\shishaohua.SHISHAOHUA1\\Downloads\\DeepFaceLab_NVIDIA\\workspace\\data_dst.mp4 -pix_fmt rgb24 ..\\..\\workspace\\data_dst\\%5d.png'
        #     # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        #     # out, err = process.communicate()
        #     # out = out.decode('cp936')
        #     # err = err.decode('cp936')
        #     # print( "%s\n" % out)
        #     # print("%s\n" % err)

    ### 从目标视频中提取图像 ###
    def extract_dst_video(self):
        input_file = self.Text_first_lineEdit.text()

        if input_file is "":
            # self.g_progress_info.append("没有设置目标视频路径，请先选择目标视频!")
            # self.NoteTipsEdit.append("没有设置目标视频路径，请先选择目标视频!")
            # self.NoteTipsEdit.append("\n没有设置目标视频路径，请先选择目标视频!")
            InfoNotifier.InfoNotifier.g_progress_info.append("没有设置目标视频路径，请先选择目标视频!")
            return

        output_dir = self.Text_Second_lineEdit.text()

        self.ori_dst_video_path = input_file
        # self.update_ui_progress_info("\n图像提取路径:" + output_dir)

        # self.update_ui_progress_info("\n图像输出路径:" + self.Text_Second_lineEdit)

        # if self.ui.rb_export_png.isChecked():
        #  output_ext = "png"
        # if self.ui.rb_export_jpg.isChecked():
        #  output_ext = "jpg"
        output_ext = "jpg"
        # self.update_ui_progress_info("\n图像格式:" + output_ext)
        # if self.setFpsEdit.text()!="Default":
        probe=ffmpeg.probe(input_file,cmd=self.ffprobe_cmd_path)
        fps_video=[]
        for stream in probe['streams']:
            fps_video.append(stream['r_frame_rate'])
            # return
        if self.setFpsEdit.text()=="Default":
            fps = 24
        else:
            fps = self.setFpsEdit.text()
        # else:
        #     info=ffmpeg.probe(input_file)
        #     vs = next(c for c in info['streams'] if c['codec_type'] == 'video')
        #     fps = vs['r_frame_rate']
        # print("fps值：",fps)
        # self.update_ui_progress_info("\n-----------------开始提取视频帧-----------------")
        # self.NoteTipsEdit.append("\n-------开始提取视频帧-------")
        InfoNotifier.InfoNotifier.g_progress_info.append("\n-------开始提取视频帧-------")
        # UIParamReflect.GlobalConfig.b_sync_block_op_in_progress = True
        QApplication.processEvents()
        self.extract_video(input_file, output_dir, output_ext, fps)
        # self.update_ui_progress_info("\n-----------------提取视频帧完毕-----------------")
        InfoNotifier.InfoNotifier.g_progress_info.append("\n-------提取视频帧完毕-------")
        ########################################################################################################output_dir提取帧路径
        # 更新一下信息
        # self.update_ui_dst_info()
        # UIParamReflect.GlobalConfig.b_sync_block_op_in_progress = False

    # def load_graph(self, graph_filename):
    #     print(graph_filename)
    #     with tf.gfile.GFile(graph_filename, 'rb') as f:
    #         graph_def = tf.GraphDef()
    #         graph_def.ParseFromString(f.read())
    #
    #     return graph_def
    # def MtcnnDectect(self):
    #     self.NoteTipsEdit.append("\n-------读取人脸信息-------")
    #     detector = MTCNN()
    #     input_dir=self.Text_Second_lineEdit.text()
    #     if not os.path.exists(input_dir):
    #         os.makedirs(input_dir)
    #
    #     img_list = glob.glob(input_dir + '/' + '*.png')
    #     img_list += glob.glob(input_dir + '/' + '*.jpg')
    #     cnt=0
    #     for img_path in img_list:
    #         image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    #         try:
    #             result = detector.detect_faces(image)
    #             # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    #
    #             bounding_box = result[0]['box']
    #             # print(result)
    #
    #             keypoints = result[0]['keypoints']
    #             land_mask_file = img_path.replace(".jpg", ".txt")
    #             file_write = open(land_mask_file, "w")
    #             x, y = keypoints['left_eye']
    #             file_write.write(f"{x} {y}\n")
    #             x, y = keypoints['right_eye']
    #             file_write.write(f"{x} {y}\n")
    #             x, y = keypoints['nose']
    #             file_write.write(f"{x} {y}\n")
    #             x, y = keypoints['mouth_left']
    #             file_write.write(f"{x} {y}\n")
    #             x, y = keypoints['mouth_right']
    #             file_write.write(f"{x} {y}\n")
    #         except:
    #             cnt += 1
    #     self.NoteTipsEdit.append("\n-------有{}张提取失败-------".format(cnt))
    # def FaceReconst(self):
    #     self.MtcnnDectect()
    #
    #     inFileDir=self.Text_Second_lineEdit.text()
    #     save_path=self.Text_Third_lineEdit.text()
    #     img_list=glob.glob(inFileDir+'/'+'*.png')
    #     img_list+=glob.glob(inFileDir+'/'+'*.jpg')
    #     # read BFM face model
    #     # transfer original BFM model to our model
    #     if not os.path.isfile('./BFM/BFM_model_front.mat'):
    #         transferBFM09()
    #
    #     # read standard landmarks for preprocessing images
    #     lm3D = load_lm3d()
    #     batchsize = 1
    #     n = 0
    #
    #     with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
    #         FaceReconstructor = Face3D()
    #         images = tf.placeholder(name='input_imgs', shape=[batchsize, 224, 224, 3], dtype=tf.float32)
    #         graph_def = self.load_graph('network/FaceReconModel.pb')
    #         tf.import_graph_def(graph_def, name='resnet', input_map={'input_imgs:0': images})
    #
    #         # output coefficients of R-Net (dim = 257)
    #         coeff = graph.get_tensor_by_name('resnet/coeff:0')
    #
    #         # reconstructing faces
    #         FaceReconstructor.Reconstruction_Block(coeff, batchsize)
    #         face_shape = FaceReconstructor.face_shape_t
    #         face_texture = FaceReconstructor.face_texture
    #         face_color = FaceReconstructor.face_color
    #         landmarks_2d = FaceReconstructor.landmark_p
    #         # recon_img = FaceReconstructor.render_imgs
    #         tri = FaceReconstructor.facemodel.face_buf
    #         with tf.Session() as sess:
    #             # print('-----------reconstructing-----------')
    #             self.NoteTipsEdit.append("\n-------正在重建-------")
    #             cnt=0
    #             for file in img_list:
    #                 try:
    #                     n+=1
    #                     print(n)
    #                     # self.NoteTipsEdit.append("\n第"+n+"张正在重建")
    #                     # load images and corresponding 5 facial landmarks
    #                     img, lm = load_img(file, file.replace('png', 'txt').replace('jpg', 'txt'))
    #                     # preprocess input image
    #                     input_img, lm_new, transform_params = Preprocess(img, lm, lm3D)
    #
    #                     # coeff_,face_shape_,face_texture_,face_color_,landmarks_2d_,recon_img_,tri_ = sess.run([coeff,\
    #                     # 	face_shape,face_texture,face_color,landmarks_2d,recon_img,tri],feed_dict = {images: input_img})
    #
    #                     coeff_, face_shape_, face_texture_, face_color_, landmarks_2d_, tri_ = sess.run([coeff, face_shape,
    #                                                                                                      face_texture,
    #                                                                                                      face_color,
    #                                                                                                      landmarks_2d, tri],
    #                                                                                                     feed_dict={
    #                                                                                                         images: input_img})
    #
    #                     # reshape outputs
    #                     input_img = np.squeeze(input_img)
    #                     face_shape_ = np.squeeze(face_shape_, (0))
    #                     face_texture_ = np.squeeze(face_texture_, (0))
    #                     face_color_ = np.squeeze(face_color_, (0))
    #                     landmarks_2d_ = np.squeeze(landmarks_2d_, (0))
    #                     # recon_img_ = np.squeeze(recon_img_, (0))
    #
    #                     # save output files
    #                     # savemat(os.path.join(save_path,file.split(os.path.sep)[-1].replace('.png','.mat').replace('jpg','mat')),{'cropped_img':input_img[:,:,::-1],'recon_img':recon_img_,'coeff':coeff_,\
    #                     # 	'face_shape':face_shape_,'face_texture':face_texture_,'face_color':face_color_,'lm_68p':landmarks_2d_,'lm_5p':lm_new})
    #                     # savemat(os.path.join(save_path,file.split(os.path.sep)[-1].replace('.png','.mat').replace('jpg','mat')),{'cropped_img':input_img[:,:,::-1],'coeff':coeff_,\
    #                     # 	'face_shape':face_shape_,'face_texture':face_texture_,'face_color':face_color_,'lm_68p':landmarks_2d_,'lm_5p':lm_new})
    #                     save_obj(
    #                         os.path.join(save_path, file.split(os.path.sep)[-1].replace('.png', '_mesh.obj').replace('jpg',
    #                                                                                                                  '_mesh.obj')),
    #                         face_shape_, tri_,
    #                         np.clip(face_color_, 0, 255) / 255)  # 3D reconstruction face (in canonical view)
    #                 except:
    #                     cnt +=1
    #             self.NoteTipsEdit.append("\n-------有{}张重建失败-------".format(cnt))
    #             self.NoteTipsEdit.append("\n-------完毕-------")

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
        # self._trigger_text.emit('\n-------读取人脸信息-------')
        InfoNotifier.InfoNotifier.g_progress_info.append('\n-------读取人脸信息-------')
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
        # self._trigger_text.emit("\n-------有{}张提取失败-------".format(cnt))
        InfoNotifier.InfoNotifier.g_progress_info.append("\n-------有{}张提取失败-------".format(cnt))
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
                # self._trigger_text.emit("\n-------正在重建-------")
                InfoNotifier.InfoNotifier.g_progress_info.append("\n-------正在重建-------")
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
                # self._trigger_text.emit("\n-------有{}张重建失败-------".format(cntt))
                # self._trigger_text.emit("\n-------完毕-------")
                InfoNotifier.InfoNotifier.g_progress_info.append("\n-------有{}张重建失败-------".format(cntt))
                InfoNotifier.InfoNotifier.g_progress_info.append("\n-------完毕-------")
                QApplication.processEvents()
                # self.Tipstext.append("\n-------有{}张重建失败-------".format(cntt))
                # self.Tipstext.append("\n-------完毕-------")


        self._trigger.emit()
    def run(self):
        self.FaceReconst_Spec_pics()

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
        # window.NoteTipsEdit.append('---------读取人脸信息-------')
        InfoNotifier.InfoNotifier.g_progress_info.append('---------读取人脸信息-------')
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
        # self._trigger_pic_text.emit("\n-------有{}张提取失败-------".format(cnt))
        InfoNotifier.InfoNotifier.g_progress_info.append("\n-------有{}张提取失败-------".format(cnt))
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
                # self._trigger_pic_text.emit("\n-------正在重建-------")
                InfoNotifier.InfoNotifier.g_progress_info.append("\n-------正在重建-------")
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
                # self._trigger_pic_text.emit("\n-------有{}张重建失败-------".format(cnt))
                # self._trigger_pic_text.emit("\n-------完毕-------")
                InfoNotifier.InfoNotifier.g_progress_info.append("\n-------有{}张重建失败-------".format(cnt))
                InfoNotifier.InfoNotifier.g_progress_info.append("\n-------完毕-------")
                # self.TextEdit.append("\n-------有{}张重建失败-------".format(cnt))
                # self.TextEdit.append("\n-------完毕-------")
                QApplication.processEvents()



        self.trigger.emit()









if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())