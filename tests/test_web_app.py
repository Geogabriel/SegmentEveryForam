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


class TestWebAppExamples(unittest.TestCase):

    def setUp(self):
        import json
        import tempfile
        from pathlib import Path
        from PIL import Image

        self.directory = tempfile.TemporaryDirectory()
        self.examples = Path(self.directory.name)
        Image.new('RGB', (300, 200), 'black').save(self.examples / 'sample.png')
        (self.examples / 'examples.json').write_text(json.dumps({'sample.png': {
            'title': 'Sample', 'scale_bar_px': [[10, 190], [110, 190]], 'scale_bar_um': 200}}))
        self.original = web_app.EXAMPLES_DIR
        web_app.EXAMPLES_DIR = self.examples

    def tearDown(self):
        web_app.EXAMPLES_DIR = self.original
        self.directory.cleanup()

    def test_masks_round_trip(self):
        polygons = [shapely.box(10, 10, 60, 60), shapely.Point(150, 100).buffer(30)]
        web_app.save_masks(polygons, self.examples / 'sample.geojson')
        loaded = web_app.load_masks(self.examples / 'sample.geojson')
        self.assertEqual(len(loaded), 2)
        self.assertTrue(loaded[0].equals(polygons[0]))

    def test_example_with_stored_masks_needs_no_models(self):
        web_app.save_masks([shapely.box(10, 10, 60, 60)], self.examples / 'sample.geojson')
        original_models = web_app.models
        web_app.models = lambda: self.fail('the models should not run')
        try:
            session, _, sample, scale, known, download, status, table = web_app.run_example(
                str(self.examples / 'sample.png'), None)
        finally:
            web_app.models = original_models
        self.assertEqual(len(session['polygons']), 1)
        self.assertEqual(session['points'], [(10, 190), (110, 190)])
        self.assertEqual(sample, 'sample')
        self.assertEqual(scale, 200)
        self.assertIn('200 µm scale bar is already marked', status)
        self.assertIn('Area (mm²)', table.value['headers'])

    def test_listed_examples_exist(self):
        web_app.EXAMPLES_DIR = self.original
        info = web_app.example_info()
        self.assertTrue(info)
        for name, details in info.items():
            self.assertTrue((self.original / name).exists(), name)
            self.assertEqual(len(details['scale_bar_px']), 2)
            self.assertGreater(details['scale_bar_um'], 0)


class TestScaleLengthWarning(unittest.TestCase):

    def session(self):
        return {'points': [(0, 0), (265, 0)]}

    def warnings(self, scale_um):
        from unittest import mock
        with mock.patch.object(web_app.gr, 'Warning') as warning:
            web_app.check_scale_length(self.session(), scale_um)
        return warning.call_count

    def test_pixel_length_warns(self):
        self.assertEqual(self.warnings(265), 1)
        self.assertEqual(self.warnings(265.4), 1)

    def test_printed_length_does_not_warn(self):
        self.assertEqual(self.warnings(500), 0)
        self.assertEqual(self.warnings(None), 0)
