# Standard imports
import math
import unittest
# Pip imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
# Local imports
import segmenteveryforam as sef
import segmenteveryforam.interactions as sfi


# Geometry calculations
def _azimuth(point1: tuple, point2: tuple) -> float:
    ''' Azimuth between 2 points (interval 0 - 180) '''
    return np.arctan2(point2[0] - point1[0], point2[1] - point1[1]) / 2


def dist(a: tuple, b: tuple) -> float:
    ''' Distance between points '''
    return math.hypot(b[0] - a[0], b[1] - a[1])


def azimuth(mrr: shapely.Polygon) -> float:
    ''' Azimuth of minimum_rotated_rectangle '''
    bbox = list(mrr.exterior.coords)
    axis1 = dist(bbox[0], bbox[3])
    axis2 = dist(bbox[0], bbox[1])
    if axis1 <= axis2:
        az = _azimuth(bbox[0], bbox[1])
    else:
        az = _azimuth(bbox[0], bbox[3])
    return az


# Comparison methods
def absolute(a, b):
    return abs(a - b)


def relative(a, b):
    return 100 * abs((a - b) / (a + b)) / 2


def absolute_angle(a, b):
    diff = absolute(a, b)
    return min(diff, np.pi - diff) * 180. / np.pi


class Comparison():
    def __init__(self, f: object, tolerance: float):
        # Comparison function
        self.f = f
        # Tolerance for comparison
        self.tolerance = tolerance


metrics = {
    'centroid-0': Comparison(absolute, 2.),
    'centroid-1': Comparison(absolute, 2.),
    'area': Comparison(relative, 2.),
    'perimeter': Comparison(relative, 2.),
    'orientation': Comparison(absolute_angle, 10.),
    'major_axis_length': Comparison(relative, 1.),
    'minor_axis_length': Comparison(relative, 2.),
    # 'max_intensity-0': Comparison(absolute, 5.),
    # 'min_intensity-0': Comparison(absolute, 10.),
    'mean_intensity-0': Comparison(relative, 2.),
    # 'max_intensity-1': Comparison(absolute, 5.),
    # 'min_intensity-1': Comparison(absolute, 10.),
    'mean_intensity-1': Comparison(relative, 2.),
    # 'max_intensity-2': Comparison(absolute, 5.),
    # 'min_intensity-2': Comparison(absolute, 10.),
    'mean_intensity-2': Comparison(relative, 2.)
}


class TestPolygonMeasurement(unittest.TestCase):
    ''' Test polygon measurement functions. '''

    @classmethod
    def setUpClass(cls):
        # Create image with random color
        cls.rgb = np.random.randint(0, 255, size=3, dtype=np.uint8)
        cls.image = np.zeros((100, 100, 3), dtype=np.uint8)
        cls.image[:, :] = cls.rgb
        # Create polygons
        cls.polygons = [shapely.Polygon(coords) for coords in [
            [[10, 10], [25, 15], [25, 25], [15, 25]],
            [[30, 30], [40, 30], [40, 40], [30, 40]]
        ]]

    def test_measure_polygon(self):
        for poly in self.polygons:
            data = sfi.measure_polygon(poly)
            # Check the centroid
            self.assertEqual(data['centroid'], poly.centroid.xy[::-1])
            # Check the area
            self.assertEqual(data['area'], poly.area)

    def test_measure_ellipse(self):
        for poly in self.polygons:
            # Get ellipse metrics
            data = sfi.measure_ellipse(sfi.measure_polygon(poly))
            # Check that the orientation matches the minimum rotated rectangle
            angle = azimuth(poly.oriented_envelope)
            self.assertEqual(data['orientation'], angle)

    def test_measure_color(self):
        for poly in self.polygons:
            # Get color metrics
            data = sfi.measure_color(self.image, poly)
            # Check the mean color intensities
            r, g, b = self.rgb
            self.assertEqual(data['mean_intensity-0'], r)
            self.assertEqual(data['mean_intensity-1'], g)
            self.assertEqual(data['mean_intensity-2'], b)


class TestGrainObject(unittest.TestCase):
    ''' Test Grain object methods. '''

    @classmethod
    def setUpClass(cls):
        # Create image with random solid color
        cls.rgb = np.random.randint(0, 255, size=3, dtype=np.uint8)
        cls.image = np.zeros((100, 100, 3), dtype=np.uint8)
        cls.image[:, :] = cls.rgb

        # Create a mock Axes object with patches
        cls.fig, cls.ax = plt.subplots()
        cls.ax.imshow(cls.image)
        cls.patches = [
            mpl.patches.Polygon(
                np.array([[10, 10], [25, 15], [25, 25], [15, 25]]), closed=True),
            mpl.patches.Polygon(
                np.array([[30, 30], [40, 30], [40, 40], [30, 40]]), closed=True)
        ]
        for patch in cls.patches:
            cls.ax.add_patch(patch)

        # Create Shapely polygons from patches
        cls.polygons = [shapely.Polygon(p.get_xy()) for p in cls.patches]

        # Create and draw Grain objects from polygons
        cls.grains = sfi.polygons_to_grains(cls.polygons, image=cls.image)
        for grain in cls.grains:
            grain.measure()
            grain.draw_patch(cls.ax)

    def test_init(self):
        # Check the number of grains
        self.assertEqual(len(self.grains), len(self.patches))
        # Check the type of grains
        for grain in self.grains:
            self.assertIsInstance(grain, sfi.Grain)

    def test_drawing(self):
        for grain, patch in zip(self.grains, self.patches):
            # Verify coordinates
            for grain_xy, patch_xy in zip(grain.patch.get_xy(), patch.get_xy()):
                self.assertTrue((grain_xy == patch_xy).all())

    def test_measure(self):
        ''' Verify grain measurements. '''
        for grain, poly in zip(self.grains, self.polygons):
            data = grain.data
            # Centroid
            centroid = poly.centroid
            self.assertEqual(data['centroid-1'], centroid.x)
            self.assertEqual(data['centroid-0'], centroid.y)
            # Area
            self.assertEqual(data['area'], poly.area)
            # Perimeter
            self.assertEqual(data['perimeter'], poly.length)
            # Orientation
            angle = azimuth(poly.oriented_envelope)
            self.assertEqual(data['orientation'], angle)
            # Color
            r, g, b = self.rgb
            self.assertEqual(data['mean_intensity-0'], r)
            self.assertEqual(data['mean_intensity-1'], g)
            self.assertEqual(data['mean_intensity-2'], b)

    def test_rescale(self):
        pass

    def test_filter_grains_by_props(self):
        small, large = sorted(self.grains, key=lambda g: g.data['area'])
        cutoff = (small.data['area'] + large.data['area']) / 2
        self.assertEqual(
            sfi.filter_grains_by_props(self.grains, area=lambda a: a > cutoff),
            [large])
        self.assertEqual(
            sfi.filter_grains_by_props(
                self.grains,
                area=lambda a: a > 0,
                perimeter=lambda p: p < small.data['perimeter'] + 1e-9),
            [small])
        self.assertEqual(sfi.filter_grains_by_props(self.grains), self.grains)


class TestGrainPlot(unittest.TestCase):
    ''' Test GrainPlot object methods. '''

    @classmethod
    def setUpClass(cls):
        cls.image = sfi.load_image('tests/fixtures/torrey_pines.jpg')
        cls.grains = sfi.load_grains(
            'tests/fixtures/torrey_pines_grains.geojson',
            image=cls.image)
        cls.plot = sfi.GrainPlot(cls.grains, cls.image)

    def test_create_plot(self):
        # Check if the plot is created
        plot = self.plot
        self.assertIsInstance(plot.fig, plt.Figure)
        self.assertIsInstance(plot.ax, plt.Axes)
        # Check if the image is displayed
        self.assertIsInstance(plot.image, np.ndarray)
        self.assertIsInstance(plot.display_image, np.ndarray)
        # Verify that image isn't downscaled
        self.assertEqual(plot.image.shape, plot.display_image.shape)

    def test_measure_methods(self):
        # Compare old/new measurement methods to ensure they reasonably agree
        for grain in self.grains:
            a = grain.measure(raster=False)
            b = grain.measure(raster=True)
            for metric, comparison in metrics.items():
                error = comparison.f(a[metric], b[metric])
                self.assertTrue(error < comparison.tolerance)


class TestDownscaledGrainPlot(unittest.TestCase):
    ''' Test GrainPlot object methods with downscaled image. '''

    @classmethod
    def setUpClass(cls):
        cls.image = sfi.load_image('tests/fixtures/torrey_pines.jpg')
        cls.grains = sfi.load_grains(
            'tests/fixtures/torrey_pines_grains.geojson',
            image=cls.image)
        cls.testdata = cls.grains[0].measure()
        cls.testxy = cls.grains[0].xy
        cls.max_dim = 320
        cls.plot = sfi.GrainPlot(
            cls.grains, cls.image, image_max_size=(cls.max_dim, cls.max_dim))

    def test_create_plot(self):
        # Check if the plot is created
        plot = self.plot
        self.assertIsInstance(plot.fig, plt.Figure)
        self.assertIsInstance(plot.ax, plt.Axes)
        # Check that display image exists and is downscaled
        self.assertIsInstance(plot.display_image, np.ndarray)
        self.assertNotEqual(plot.image.shape, plot.display_image.shape)
        self.assertEqual(max(plot.display_image.shape), self.max_dim)
        # Check that grain sizes are downscaled
        pass

    def test_measurement(self):
        # Ensure measurements are correctly calculated
        for grain in self.grains:
            data = grain.data
            self.assertIn('area', data)
            self.assertIn('perimeter', data)
            self.assertIn('centroid-0', data)
            self.assertIn('centroid-1', data)

    def test_grain_scale(self):
        # Verify grain coordinates
        testdata = self.testdata
        graindata = self.grains[0].data
        self.assertEqual(testdata['area'], graindata['area'])
        plotdata = self.plot.grains[0].data
        self.assertEqual(testdata['area'], plotdata['area'])
        # Verify grain data
        testxy = self.testxy
        grainxy = self.grains[0].xy
        self.assertTrue((testxy == grainxy).all())
        plotxy = self.plot.grains[0].xy
        self.assertTrue((testxy == plotxy).all())


class TestUndo(unittest.TestCase):
    ''' Undo restores grains after add, delete, and merge edits. '''

    def setUp(self):
        self.image = np.zeros((200, 200, 3), dtype=np.uint8)
        squares = [shapely.box(10, 10, 50, 50), shapely.box(40, 10, 80, 50),
                   shapely.box(120, 120, 160, 160)]
        self.grains = [sfi.Grain(np.array(sq.exterior.xy), self.image) for sq in squares]
        self.plot = sfi.ForamPlot(list(self.grains), self.image, blit=False)

    def tearDown(self):
        plt.close(self.plot.fig)

    def select(self, *grains):
        for grain in grains:
            grain.select()
        self.plot.selected_grains = list(grains)

    def test_nothing_to_undo(self):
        self.assertFalse(self.plot.undo())
        self.assertEqual(self.plot.grains, self.grains)

    def test_undo_delete_restores_order(self):
        self.select(self.grains[0], self.grains[1])
        self.plot.delete_grains()
        self.assertEqual(self.plot.grains, [self.grains[2]])
        self.assertTrue(self.plot.undo())
        self.assertEqual(self.plot.grains, self.grains)
        for grain in self.plot.grains:
            self.assertFalse(grain.selected)
            self.assertIn(grain.patch, self.plot.ax.patches)

    def test_undo_merge(self):
        self.select(self.grains[0], self.grains[1])
        merged = self.plot.merge_grains()
        self.assertEqual(self.plot.grains, [self.grains[2], merged])
        self.assertEqual(len(self.plot.history), 1)
        self.plot.undo()
        self.assertEqual(self.plot.grains, self.grains)
        self.assertNotIn(merged.patch, self.plot.ax.patches)

    def test_undo_create(self):
        original = sfi.predict_from_prompts
        sfi.predict_from_prompts = lambda **kwargs: shapely.box(100, 20, 140, 60).exterior.xy
        try:
            self.plot.set_point((120, 40))
            new_grain = self.plot.create_grain()
        finally:
            sfi.predict_from_prompts = original
        self.assertIs(self.plot.grains[-1], new_grain)
        self.plot.undo()
        self.assertEqual(self.plot.grains, self.grains)

    def test_undo_steps_back_in_order(self):
        self.select(self.grains[2])
        self.plot.delete_grains()
        self.select(self.grains[0], self.grains[1])
        merged = self.plot.merge_grains()
        self.plot.undo()
        self.assertEqual(self.plot.grains, self.grains[:2])
        self.plot.undo()
        self.assertEqual(self.plot.grains, self.grains)
        self.assertFalse(self.plot.undo())
        self.assertIsNotNone(merged)


if __name__ == '__main__':
    unittest.main()
