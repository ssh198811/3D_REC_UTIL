
# -*- mode: python -*-
 
block_cipher = None
#data_map={("mtcnn","mtcnn"),("ffmpeg/ffmpeg.exe","ffmpeg"),("ffmpeg/ffprobe.exe","ffmpeg"),
a=Analysis(['ui.py'],
            pathex=['E:\\PycharmProject\\mtcnn\\Three_D_Rec'],
            binaries=[],
            datas=[],
            hiddenimports=[],
            hookspath=[],
            runtime_hooks=[],
            excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='Three_D_Rec',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False)
          
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='FaceReconstruction')