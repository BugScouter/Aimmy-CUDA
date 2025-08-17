using Aimmy2.Class;
using Class;
using System.Drawing;
using System.Windows;
using Visuality;

namespace Aimmy2.AILogic
{
    public class OverlayManager
    {
        private readonly DetectedPlayerWindow _detectedPlayerOverlay;
        public OverlayManager(DetectedPlayerWindow overlay)
        {
            _detectedPlayerOverlay = overlay;
        }
        public async void UpdateFOV()
        {
            if (Dictionary.dropdownState["Detection Area Type"] == "Closest to Mouse" && Dictionary.toggleState["FOV"])
            {
                var mousePosition = WinAPICaller.GetCursorPosition();

                // Check if mouse is on the current display
                if (!DisplayManager.IsPointInCurrentDisplay(new System.Windows.Point(mousePosition.X, mousePosition.Y)))
                {
                    // Mouse is on a different display - don't update FOV position
                    return;
                }

                // Translate mouse position relative to current display
                var displayRelativeX = mousePosition.X - DisplayManager.ScreenLeft;
                var displayRelativeY = mousePosition.Y - DisplayManager.ScreenTop;

                await Application.Current.Dispatcher.BeginInvoke(() =>
                    Dictionary.FOVWindow.FOVStrictEnclosure.Margin = new Thickness(
                        Convert.ToInt16(displayRelativeX / WinAPICaller.scalingFactorX) - 320, // this is based off the window size, not the size of the model -whip
                        Convert.ToInt16(displayRelativeY / WinAPICaller.scalingFactorY) - 320, 0, 0));
            }
        }
        public void UpdateOverlay(Prediction closestPrediction, RectangleF LastDetectionBox, double AIConf)
        {
            var scalingFactorX = WinAPICaller.scalingFactorX;
            var scalingFactorY = WinAPICaller.scalingFactorY;

            // Convert screen coordinates to display-relative coordinates
            var displayRelativeX = LastDetectionBox.X - DisplayManager.ScreenLeft;
            var displayRelativeY = LastDetectionBox.Y - DisplayManager.ScreenTop;

            // Calculate center position in display-relative coordinates
            var centerX = Convert.ToInt16(displayRelativeX / scalingFactorX) + (LastDetectionBox.Width / 2.0);
            var centerY = Convert.ToInt16(displayRelativeY / scalingFactorY);

            Application.Current.Dispatcher.Invoke(() =>
            {
                if (Dictionary.toggleState["Show AI Confidence"])
                {
                    _detectedPlayerOverlay.DetectedPlayerConfidence.Opacity = 1;
                    _detectedPlayerOverlay.DetectedPlayerConfidence.Content = $"{closestPrediction.ClassName}: {Math.Round((AIConf * 100), 2)}%";

                    var labelEstimatedHalfWidth = _detectedPlayerOverlay.DetectedPlayerConfidence.ActualWidth / 2.0;
                    _detectedPlayerOverlay.DetectedPlayerConfidence.Margin = new Thickness(
                        centerX - labelEstimatedHalfWidth,
                        centerY - _detectedPlayerOverlay.DetectedPlayerConfidence.ActualHeight - 2, 0, 0);
                }
                var showTracers = Dictionary.toggleState["Show Tracers"];
                _detectedPlayerOverlay.DetectedTracers.Opacity = showTracers ? 1 : 0;
                if (showTracers)
                {
                    var tracerPosition = Dictionary.dropdownState["Tracer Position"];

                    var boxTop = centerY;
                    var boxBottom = centerY + LastDetectionBox.Height;
                    var boxHorizontalCenter = centerX;
                    var boxVerticalCenter = centerY + (LastDetectionBox.Height / 2.0);
                    var boxLeft = centerX - (LastDetectionBox.Width / 2.0);
                    var boxRight = centerX + (LastDetectionBox.Width / 2.0);

                    switch (tracerPosition)
                    {
                        case "Top":
                            _detectedPlayerOverlay.DetectedTracers.X2 = boxHorizontalCenter;
                            _detectedPlayerOverlay.DetectedTracers.Y2 = boxTop;
                            break;

                        case "Bottom":
                            _detectedPlayerOverlay.DetectedTracers.X2 = boxHorizontalCenter;
                            _detectedPlayerOverlay.DetectedTracers.Y2 = boxBottom;
                            break;

                        case "Middle":
                            var screenHorizontalCenter = DisplayManager.ScreenWidth / (2.0 * WinAPICaller.scalingFactorX);
                            if (boxHorizontalCenter < screenHorizontalCenter)
                            {
                                // if the box is on the left half of the screen, aim for the right-middle of the box
                                _detectedPlayerOverlay.DetectedTracers.X2 = boxRight;
                                _detectedPlayerOverlay.DetectedTracers.Y2 = boxVerticalCenter;
                            }
                            else
                            {
                                // if the box is on the right half, aim for the left-middle
                                _detectedPlayerOverlay.DetectedTracers.X2 = boxLeft;
                                _detectedPlayerOverlay.DetectedTracers.Y2 = boxVerticalCenter;
                            }
                            break;

                        default:
                            // default to the bottom-center if the setting is unrecognized
                            _detectedPlayerOverlay.DetectedTracers.X2 = boxHorizontalCenter;
                            _detectedPlayerOverlay.DetectedTracers.Y2 = boxBottom;
                            break;
                    }
                }

                _detectedPlayerOverlay.Opacity = Dictionary.sliderSettings["Opacity"];

                _detectedPlayerOverlay.DetectedPlayerFocus.Opacity = 1;
                _detectedPlayerOverlay.DetectedPlayerFocus.Margin = new Thickness(
                    centerX - (LastDetectionBox.Width / 2.0), centerY, 0, 0);
                _detectedPlayerOverlay.DetectedPlayerFocus.Width = LastDetectionBox.Width;
                _detectedPlayerOverlay.DetectedPlayerFocus.Height = LastDetectionBox.Height;
            });
        }

        public void DisableOverlay(DetectedPlayerWindow DetectedPlayerOverlay)
        {
            if (Dictionary.toggleState["Show Detected Player"] && Dictionary.DetectedPlayerOverlay != null)
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    if (Dictionary.toggleState["Show AI Confidence"])
                    {
                        DetectedPlayerOverlay!.DetectedPlayerConfidence.Opacity = 0;
                    }

                    if (Dictionary.toggleState["Show Tracers"])
                    {
                        DetectedPlayerOverlay!.DetectedTracers.Opacity = 0;
                    }

                    DetectedPlayerOverlay!.DetectedPlayerFocus.Opacity = 0;
                });
            }
        }

    }
}
