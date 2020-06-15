import subprocess
import numpy as np
import ffmpeg
from pathlib import Path
from core import pathex
from core.interact import interact as io
# from ui import UIParamReflect
# from ui import InfoNotifier
from PyQt5.QtWidgets import QApplication

def extract_video(input_file, output_dir, output_ext=None, fps=None):
    input_file_path = Path(input_file)
    output_path = Path(output_dir)

    if not output_path.exists():
        output_path.mkdir(exist_ok=True)
    InfoNotifier.InfoNotifier.g_progress_info.append("\n视频帧输出目录: " + str(Path(output_path).absolute()))

    if input_file_path.suffix == '.*':
        input_file_path = pathex.get_first_file_by_stem (input_file_path.parent, input_file_path.stem)
    else:
        if not input_file_path.exists():
            input_file_path = None

    InfoNotifier.InfoNotifier.g_progress_info.append("\n视频输入路径:"+str(input_file_path))

    if input_file_path is None:
        io.log_err("input_file not found.")
        InfoNotifier.InfoNotifier.g_progress_info.append("\n视频输入路径不存在")
        return

    if fps is None:
        fps = io.input_int ("Enter FPS", 0, help_message="How many frames of every second of the video will be extracted. 0 - full fps")
    InfoNotifier.InfoNotifier.g_progress_info.append("\n视频帧抽取频率: full fps" )

    if output_ext is None:
        output_ext = io.input_str ("Output image format", "png", ["png","jpg"], help_message="png is lossless, but extraction is x10 slower for HDD, requires x10 more disk space than jpg.")

    InfoNotifier.InfoNotifier.g_progress_info.append("\n视频帧输出格式频率: " + output_ext)

    filenames = pathex.get_image_paths (output_path, ['.'+output_ext])
    if len(filenames) != 0:
        InfoNotifier.InfoNotifier.g_progress_info.append("\n视频帧输出目录不为空, 该目录将被清空!")


    for filename in filenames:
        Path(filename).unlink()
        QApplication.processEvents()


    job = ffmpeg.input(str(input_file_path))

    kwargs = {'pix_fmt': 'rgb24'}
    if fps != 0:
        kwargs.update ({'r':str(fps)})

    if output_ext == 'jpg':
        kwargs.update ({'q:v':'2'}) #highest quality for jpg

    job = job.output( str (output_path / ('%5d.'+output_ext)), **kwargs )

    try:
        job, err = job.run(cmd = UIParamReflect.GlobalConfig.ffmpeg_cmd_path)
    except:
        io.log_err ("ffmpeg fail, job commandline:" + str(job.compile()) )

    # cmd = 'E:\\Users\\shishaohua.SHISHAOHUA1\\Downloads\\DeepFaceLab_NVIDIA\\_internal\\ffmpeg\\ffmpeg -i E:\\Users\\shishaohua.SHISHAOHUA1\\Downloads\\DeepFaceLab_NVIDIA\\workspace\\data_dst.mp4 -pix_fmt rgb24 ..\\..\\workspace\\data_dst\\%5d.png'
    #     # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    #     # out, err = process.communicate()
    #     # out = out.decode('cp936')
    #     # err = err.decode('cp936')
    #     # print( "%s\n" % out)
    #     # print("%s\n" % err)



def cut_video ( input_file, from_time=None, to_time=None, audio_track_id=None, bitrate=None):
    input_file_path = Path(input_file)
    if input_file_path is None:
        io.log_err("input_file not found.")
        return

    output_file_path = input_file_path.parent / (input_file_path.stem + "_cut" + input_file_path.suffix)

    if from_time is None:
        from_time = io.input_str ("From time", "00:00:00.000")

    if to_time is None:
        to_time = io.input_str ("To time", "00:00:00.000")

    if audio_track_id is None:
        audio_track_id = io.input_int ("Specify audio track id.", 0)

    if bitrate is None:
        bitrate = max (1, io.input_int ("Bitrate of output file in MB/s", 25) )

    kwargs = {"c:v": "libx264",
              "b:v": "%dM" %(bitrate),
              "pix_fmt": "yuv420p",
             }

    job = ffmpeg.input(str(input_file_path), ss=from_time, to=to_time)

    job_v = job['v:0']
    job_a = job['a:' + str(audio_track_id) + '?' ]

    job = ffmpeg.output(job_v, job_a, str(output_file_path), **kwargs).overwrite_output()

    try:
        job = job.run()
    except:
        io.log_err ("ffmpeg fail, job commandline:" + str(job.compile()) )

def denoise_image_sequence( input_dir, ext=None, factor=None ):
    input_path = Path(input_dir)

    if not input_path.exists():
        io.log_err("input_dir not found.")
        return

    if ext is None:
        ext = io.input_str ("Input image format (extension)", "png")

    if factor is None:
        factor = np.clip ( io.input_int ("Denoise factor?", 5, add_info="1-20"), 1, 20 )

    kwargs = {}
    if ext == 'jpg':
        kwargs.update ({'q:v':'2'})

    job = ( ffmpeg
            .input(str ( input_path / ('%5d.'+ext) ) )
            .filter("hqdn3d", factor, factor, 5,5)
            .output(str ( input_path / ('%5d.'+ext) ), **kwargs )
           )

    try:
        job = job.run()
    except:
        io.log_err ("ffmpeg fail, job commandline:" + str(job.compile()) )

def video_from_sequence( input_dir, output_file, reference_file=None, ext=None, fps=None, bitrate=None, include_audio=False, lossless=None ):
    input_path = Path(input_dir)
    output_file_path = Path(output_file)
    reference_file_path = Path(reference_file) if reference_file is not None else None

    if not input_path.exists():
        io.log_err("input_dir not found.")
        return

    if not output_file_path.parent.exists():
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        return

    out_ext = output_file_path.suffix

    if ext is None:
        ext = io.input_str ("Input image format (extension)", "png")

    if lossless is None:
        lossless = io.input_bool ("Use lossless codec", False)

    video_id = None
    audio_id = None
    ref_in_a = None
    if reference_file_path is not None:
        if reference_file_path.suffix == '.*':
            reference_file_path = pathex.get_first_file_by_stem (reference_file_path.parent, reference_file_path.stem)
        else:
            if not reference_file_path.exists():
                reference_file_path = None

        if reference_file_path is None:
            io.log_err("reference_file not found.")
            return

        #probing reference file
        probe = ffmpeg.probe (str(reference_file_path), cmd = UIParamReflect.GlobalConfig.ffprobe_cmd_path)

        #getting first video and audio streams id with fps
        for stream in probe['streams']:
            if video_id is None and stream['codec_type'] == 'video':
                video_id = stream['index']
                fps = stream['r_frame_rate']

            if audio_id is None and stream['codec_type'] == 'audio':
                audio_id = stream['index']

        if audio_id is not None:
            #has audio track
            ref_in_a = ffmpeg.input (str(reference_file_path))[str(audio_id)]

    if fps is None:
        #if fps not specified and not overwritten by reference-file
        #ps = max (1, io.input_int ("Enter FPS", 25) )
        fps = UIParamReflect.UIParam2Config.default_fps
        InfoNotifier.InfoNotifier.g_progress_info.append(f"默认合成帧率：{fps}")

    if not lossless and bitrate is None:
        # bitrate = max (1, io.input_int ("Bitrate of output file in MB/s", 16) )
        bitrate = UIParamReflect.UIParam2Config.bit_rate
        io.log_info("Bitrate of output file in MB/s " + str(bitrate))

    input_image_paths = pathex.get_image_paths(input_path)

    i_in = ffmpeg.input('pipe:', format='image2pipe', r=fps)

    output_args = [i_in]

    if include_audio and ref_in_a is not None:
        output_args += [ref_in_a]

    output_args += [str (output_file_path)]

    output_kwargs = {}

    if lossless:
        output_kwargs.update ({"c:v": "libx264",
                               "crf": "0",
                               "pix_fmt": "yuv420p",
                              })
    else:
        output_kwargs.update ({"c:v": "libx264",
                               "b:v": "%dM" %(bitrate),
                               "pix_fmt": "yuv420p",
                              })
                              
    if include_audio and ref_in_a is not None:
        output_kwargs.update ({"c:a": "aac",
                               "b:a": "192k",
                               "ar" : "48000"
                               })

    job = ( ffmpeg.output(*output_args, **output_kwargs).overwrite_output() )

    try:
        job_run = job.run_async(pipe_stdin=True, cmd = UIParamReflect.GlobalConfig.ffmpeg_cmd_path)

        for image_path in input_image_paths:
            with open (image_path, "rb") as f:
                image_bytes = f.read()
                job_run.stdin.write (image_bytes)

        job_run.stdin.close()
        job_run.wait()
    except:
        io.log_err ("ffmpeg fail, job commandline:" + str(job.compile()) )
