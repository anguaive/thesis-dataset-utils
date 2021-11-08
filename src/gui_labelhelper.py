#!/usr/bin/python3

from PyQt5.QtWidgets import QPushButton, QMessageBox, QApplication, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QLineEdit, QFileDialog, QGroupBox
from PyQt5.QtCore import Qt, pyqtSignal, QRect
from PyQt5.QtGui import QPixmap
import sys
import os
from collections import deque
from pathlib import Path
from utils import trays_path

# FIXME(rg): wrong labels when undoing across part folders or trays
# FIXME(rg): restoring (undoing) the last item in the category removes the category folder instead of keeping it empty
# TODO(rg): image scaling - scale up small images somewhat and fit huge images to window size

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
trays = list(trays_path.glob('*'))
trays.sort(reverse = True)

class HotkeyButton(QPushButton):
    def __init__(self, hotkey, handler):
        super().__init__(hotkey)
        self.hotkey = hotkey
        self.handler = handler
        self.clicked.connect(self.handleClick)

    def handleClick(self):
        self.handler(self.hotkey)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.part_folders = []
        self.images = []
        self.current_image = None
        self.undo_buffer = deque(maxlen=20)

        self.current_tray_label = QLabel()
        self.current_part_folder_label = QLabel()
        self.current_image_label = QLabel()
        self.current_image_image = QLabel()
        self.undo_button = QPushButton('H')

        layout = QHBoxLayout()
        image_layout = QVBoxLayout()
        image_layout.setAlignment(Qt.AlignTop)
        image_layout.addWidget(self.current_tray_label)
        image_layout.addWidget(self.current_part_folder_label)
        image_layout.addWidget(self.current_image_label)
        image_layout.addWidget(self.current_image_image)
        layout.addLayout(image_layout)

        buttons_layout = QVBoxLayout()
        buttons_layout.setAlignment(Qt.AlignCenter)
        mark_present_dirty_layout = QHBoxLayout()
        mark_present_dirty_button = HotkeyButton('U', self.input_handler)
        mark_present_dirty_button.setShortcut('U')
        mark_present_dirty_label = QLabel('Mark as present+dirty')
        mark_present_dirty_layout.addWidget(mark_present_dirty_button)
        mark_present_dirty_layout.addWidget(mark_present_dirty_label)
        buttons_layout.addLayout(mark_present_dirty_layout)

        mark_present_clean_layout = QHBoxLayout()
        mark_present_clean_button = HotkeyButton('J', self.input_handler)
        mark_present_clean_button.setShortcut('J')
        mark_present_clean_label = QLabel('Mark as present+clean')
        mark_present_clean_layout.addWidget(mark_present_clean_button)
        mark_present_clean_layout.addWidget(mark_present_clean_label)
        buttons_layout.addLayout(mark_present_clean_layout)

        mark_missing_dirty_layout = QHBoxLayout()
        mark_missing_dirty_button = HotkeyButton('I', self.input_handler)
        mark_missing_dirty_button.setShortcut('I')
        mark_missing_dirty_label = QLabel('Mark as missing+dirty')
        mark_missing_dirty_layout.addWidget(mark_missing_dirty_button)
        mark_missing_dirty_layout.addWidget(mark_missing_dirty_label)
        buttons_layout.addLayout(mark_missing_dirty_layout)

        mark_missing_clean_layout = QHBoxLayout()
        mark_missing_clean_button = HotkeyButton('K', self.input_handler)
        mark_missing_clean_button.setShortcut('K')
        mark_missing_clean_label = QLabel('Mark as missing+clean')
        mark_missing_clean_layout.addWidget(mark_missing_clean_button)
        mark_missing_clean_layout.addWidget(mark_missing_clean_label)
        buttons_layout.addLayout(mark_missing_clean_layout)

        mark_uncertain_layout = QHBoxLayout()
        mark_uncertain_button = HotkeyButton('O', self.input_handler)
        mark_uncertain_button.setShortcut('O')
        mark_uncertain_label = QLabel('Mark as uncertain')
        mark_uncertain_layout.addWidget(mark_uncertain_button)
        mark_uncertain_layout.addWidget(mark_uncertain_label)
        buttons_layout.addLayout(mark_uncertain_layout)

        undo_layout = QHBoxLayout()
        self.undo_button.setShortcut('H')
        self.undo_button.setDisabled(True)
        self.undo_button.clicked.connect(self.undo_handler)
        undo_label = QLabel('Undo')
        undo_layout.addWidget(self.undo_button)
        undo_layout.addWidget(undo_label)
        buttons_layout.addLayout(undo_layout)

        layout.addLayout(buttons_layout)
        self.setLayout(layout)

        self.next()

    def next(self):
        if len(self.part_folders) == 0 and len(self.images) == 0:
            if len(trays) == 0:
                print('Done; no trays left')
                exit(0)

            tray = trays.pop()
            self.current_tray_label.setText(f'<b>Tray:</b> <code>{tray.name}</code> <i>({len(trays)} remaining)</i>')
            self.part_folders = list((tray / 'part_images').glob('*'))
            self.part_folders.sort(reverse = True, key = lambda p: int(p.name))

        if len(self.images) == 0:
            if len(self.part_folders) == 0:
                self.next()
                return

            part_folder = self.part_folders.pop()
            self.current_part_folder_label.setText(f'<b>Part:</b> <code>{part_folder.name}</code> <i>({len(self.part_folders)} remaining</i>)')
            self.images = list(part_folder.glob('*.png'))
            self.images.sort(reverse = True)
            
            try:
                os.mkdir(part_folder / 'present')
                os.mkdir(part_folder / 'present' / 'dirty')
                os.mkdir(part_folder / 'present' / 'clean')
                os.mkdir(part_folder / 'missing')
                os.mkdir(part_folder / 'missing' / 'dirty')
                os.mkdir(part_folder / 'missing' / 'clean')
                os.mkdir(part_folder / 'uncertain')
                os.mkdir(part_folder / 'uncertain' / 'dirty')
            except FileExistsError:
                pass

        if len(self.images) == 0:
            self.next()
            return 

        self.current_image = self.images.pop()
        self.current_image_label.setText(f'<b>Image:</b> <code>{self.current_image.name}</code> <i>({len(self.images)} remaining</i>)')
        self.current_image_image.setPixmap(QPixmap(str(self.current_image)))

    def input_handler(self, key):
        if key == 'U':
            folder = 'present/dirty'
        elif key == 'J':
            folder = 'present/clean'
        elif key == 'I':
            folder = 'missing/dirty'
        elif key == 'K':
            folder = 'missing/clean'
        elif key == 'O':
            folder = 'uncertain/dirty'
        else:
            return

        new_path = self.current_image.parent / folder / self.current_image.name
        os.rename(self.current_image, new_path)
        self.undo_buffer.append(new_path)
        self.undo_button.setDisabled(False)

        self.next()

    def undo_handler(self):
        self.images.append(self.current_image)
        path = self.undo_buffer.pop()
        old_path = path.parents[2] / path.name
        os.rename(path, old_path)
        self.current_image = old_path
        self.current_image_label.setText(f'<b>Image:</b> <code>{old_path.name}</code> <i>({len(self.images)} remaining</i>)')
        self.current_image_image.setPixmap(QPixmap(str(old_path)))
        if len(self.undo_buffer) == 0:
            self.undo_button.setDisabled(True)

app = QApplication([])

# The default font size is way too tiny (at least on my screen)
font = app.font()
font.setPointSize(12)
app.setFont(font)
app.setStyleSheet("QPushButton { max-width: 30px }");

mainWindow = MainWindow()

# I needed a custom window title to set up an exception in my window manager
mainWindow.setWindowTitle('labeler')
mainWindow.setFixedSize(800, 600)
mainWindow.show()

sys.exit(app.exec_())
