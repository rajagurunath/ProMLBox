# -*- mode: python -*-

block_cipher = None


a = Analysis(['Promlbox.py'],
             pathex=['D:\\dummies\\personal assistant\\proml'],
             binaries=[],
             datas=[],
             hiddenimports=['cython', 'sklearn', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.ensemble', 'sklearn.tree._utils'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='Promlbox',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
