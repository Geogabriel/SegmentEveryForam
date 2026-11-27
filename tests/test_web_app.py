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
        session, *_ = web_app.edit(self.session, web_app.REMOVE, None, None, click(20, 20))
        self.assertEqual(session['polygons'], self.squares[1:])
        session, _, download, status, _ = web_app.undo(session)
        self.assertEqual(session['polygons'], self.squares)
        self.assertIsNone(download)
        self.assertIn('2 specimens', status)

    def test_undo_scale_bar_points(self):
        session, *_ = web_app.edit(self.session, web_app.CALIBRATE, None, None, click(5, 190))
        session, *_ = web_app.edit(session, web_app.CALIBRATE, None, None, click(105, 190))
        self.assertEqual(len(session['points']), 2)
        web_app.undo(session)
        self.assertEqual(session['points'], [(5, 190)])
        web_app.undo(session)
        self.assertEqual(session['points'], [])

    def test_failed_click_is_not_recorded(self):
        with self.assertRaises(gr.Error):
            web_app.edit(self.session, web_app.REMOVE, None, None, click(80, 80))
        self.assertEqual(self.session['history'], [])
        with self.assertRaises(gr.Error):
            web_app.undo(self.session)

    def test_history_is_limited(self):
        for i in range(web_app.UNDO_LIMIT + 5):
            web_app.edit(self.session, web_app.CALIBRATE, None, None, click(i, 0))
        self.assertEqual(len(self.session['history']), web_app.UNDO_LIMIT)

    def test_review_mode_ignores_clicks(self):
        result = web_app.edit(self.session, web_app.REVIEW, None, None, click(20, 20))
        self.assertEqual(self.session['history'], [])
        self.assertEqual(len(result), 5)



class TestWebAppMeasurementTable(unittest.TestCase):

    def setUp(self):
        image = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
        self.session = {'image': image, 'name': 'test.png',
                        'polygons': [shapely.box(10, 10, 60, 60), shapely.Point(200, 200).buffer(40)],
                        'points': [], 'history': [], 'selected': None, 'directory': None}

    def rows(self, table):
        return table.value['headers'], table.value['data']

    def test_pixels_before_calibration(self):
        headers, data = self.rows(web_app.measurement_table(self.session))
        self.assertIn('Area (px²)', headers)
        self.assertEqual([row[0] for row in data], [1, 2])
        self.assertAlmostEqual(data[0][headers.index('Area (px²)')], 2500, delta=1)

    def test_millimetres_from_scale_bar(self):
        # 100 px = 500 µm, so the 50 px square is 0.25 mm wide.
        self.session['points'] = [(0, 290), (100, 290)]
        table = web_app.measurement_table(self.session, scale_um=500)
        headers, data = self.rows(table)
        self.assertIn('0.2 px per µm', table.label)
        self.assertAlmostEqual(data[0][headers.index('Area (mm²)')], 0.0625, places=3)

    def test_known_calibration_wins(self):
        self.session['points'] = [(0, 290), (100, 290)]
        table = web_app.measurement_table(self.session, scale_um=500, pixels_per_meter=1e6)
        self.assertIn('1 px per µm', table.label)

    def test_click_highlights_specimen(self):
        session, _, download, status, table = web_app.edit(
            self.session, web_app.REVIEW, None, None, click(200, 200))
        self.assertEqual(session['selected'], 1)
        self.assertIn('Specimen 2', status)
        self.assertIn('background-color', table.value['metadata']['styling'][1][0])
        self.assertEqual(table.value['metadata']['styling'][0][0], '')
        web_app.edit(self.session, web_app.REVIEW, None, None, click(280, 20))
        self.assertIsNone(self.session['selected'])

    def test_row_click_highlights_specimen(self):
        session, *_ = web_app.select_row(self.session, None, None, click(0, 3))
        self.assertEqual(session['selected'], 0)

    def test_table_follows_edits(self):
        session, *_, table = web_app.edit(self.session, web_app.REMOVE, None, None, click(20, 20))
        self.assertEqual(len(table.value['data']), 1)
        *_, table = web_app.undo(session)
        self.assertEqual(len(table.value['data']), 2)

    def test_empty_table(self):
        self.session['polygons'] = []
        headers, data = self.rows(web_app.measurement_table(self.session))
        self.assertEqual(headers[0], '#')


if __name__ == '__main__':
    unittest.main()
