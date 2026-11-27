''' Tests for the web app's review edits that don't need the models. '''
import types
import unittest

import numpy as np
import pytest
import shapely

gr = pytest.importorskip('gradio')
import web_app  # noqa: E402


def click(x, y):
    return types.SimpleNamespace(index=(x, y))


class TestWebAppUndo(unittest.TestCase):

    def setUp(self):
        self.squares = [shapely.box(10, 10, 50, 50), shapely.box(100, 100, 150, 150)]
        self.session = {'image': np.zeros((200, 200, 3), dtype=np.uint8), 'name': 'test.png',
                        'polygons': list(self.squares), 'points': [], 'history': [],
                        'directory': None}

    def test_undo_remove(self):
        session, *_ = web_app.edit(self.session, web_app.REMOVE, click(20, 20))
        self.assertEqual(session['polygons'], self.squares[1:])
        session, _, download, status = web_app.undo(session)
        self.assertEqual(session['polygons'], self.squares)
        self.assertIsNone(download)
        self.assertIn('2 specimens', status)

    def test_undo_scale_bar_points(self):
        session, *_ = web_app.edit(self.session, web_app.CALIBRATE, click(5, 190))
        session, *_ = web_app.edit(session, web_app.CALIBRATE, click(105, 190))
        self.assertEqual(len(session['points']), 2)
        web_app.undo(session)
        self.assertEqual(session['points'], [(5, 190)])
        web_app.undo(session)
        self.assertEqual(session['points'], [])

    def test_failed_click_is_not_recorded(self):
        with self.assertRaises(gr.Error):
            web_app.edit(self.session, web_app.REMOVE, click(80, 80))
        self.assertEqual(self.session['history'], [])
        with self.assertRaises(gr.Error):
            web_app.undo(self.session)

    def test_history_is_limited(self):
        for i in range(web_app.UNDO_LIMIT + 5):
            web_app.edit(self.session, web_app.CALIBRATE, click(i, 0))
        self.assertEqual(len(self.session['history']), web_app.UNDO_LIMIT)

    def test_review_mode_ignores_clicks(self):
        result = web_app.edit(self.session, web_app.REVIEW, click(20, 20))
        self.assertEqual(self.session['history'], [])
        self.assertEqual(len(result), 4)


if __name__ == '__main__':
    unittest.main()
