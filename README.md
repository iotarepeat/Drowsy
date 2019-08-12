# Drowsy
Sleep and fatigue detection using Computer Vision

There are two tests:
1. Eye test:
  Calculates EAR (Eye Aspect Ratio) to detect if eyes are closed
 
2. Yawn test:
  Similarly detects if mouth is open 
 
 Both of these tests will run for every frame.
 If either of tests results positive for MAX_FRAMES times a beep is played
