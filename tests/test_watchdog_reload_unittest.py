import unittest

from werkzeugkoppler.config_reload import watchdog_path_matches_config


class WatchdogPathMatchTests(unittest.TestCase):
    def test_matches_same_filename_for_absolute_path(self) -> None:
        self.assertTrue(watchdog_path_matches_config("/tmp/work/config.yaml", "config.yaml"))

    def test_matches_same_filename_for_relative_path(self) -> None:
        self.assertTrue(watchdog_path_matches_config("config.yaml", "config.yaml"))

    def test_does_not_match_different_filename(self) -> None:
        self.assertFalse(watchdog_path_matches_config("/tmp/work/other.yaml", "config.yaml"))

    def test_does_not_match_none(self) -> None:
        self.assertFalse(watchdog_path_matches_config(None, "config.yaml"))


if __name__ == "__main__":
    unittest.main()
