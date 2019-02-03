package team6072.vision;

import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.vision.*;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

import edu.wpi.cscore.CvSource;
import edu.wpi.cscore.VideoSource;
import edu.wpi.first.cameraserver.CameraServer;
import edu.wpi.first.networktables.*;
import org.opencv.core.MatOfPoint2f;

import org.opencv.imgproc.Imgproc;

import org.opencv.core.Point;
import java.util.ArrayList;
import org.opencv.core.Mat;

import java.util.List;

import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;

public class CloseUpPipelineListener implements VisionRunner.Listener<CloseUpPipeline> {

    // The object to synchronize on to make sure the vision thread doesn't
    // write to variables the main thread is using.
    private final Object visionLock = new Object();

    // Network Table Entrys
    private NetworkTable mTbl;
    private NetworkTableEntry mX1;
    private NetworkTableEntry mY1;
    private NetworkTableEntry mX2;
    private NetworkTableEntry mY2;

    private final double TAPE_DIST_FROM_CENTER_INCHES_X = 5.65572;
    private final double HALF_OF_CAMERA_FOV_ANGLE_X = .433304;
    private final int HALF_OF_CAMERA_PIXEL_WIDTH_PIXELS = 80;    

    // Pipelines
    private CloseUpPipeline mPipeline;

    // Camera Servers
    private CameraServer mCameraServer;
    private CvSource mHVSThresholdOutput;
    private CvSource mRectanglesOutput;

    // Counters
    private int mCallCounter = 0;
    private int mCounter;

    public double abs(double num) {
        if (num < 0) {
            num = num * -1;
        }
        return num;
    }

    public CloseUpPipelineListener() {
        // instantiate Network Tables
        NetworkTableInstance tblInst = NetworkTableInstance.getDefault();
        mTbl = tblInst.getTable("vision");
        // Instantiate Camera Server Stuff
        mCameraServer = CameraServer.getInstance();
        // output HSVThreshold output to the camera server on Shuffleboard
        mHVSThresholdOutput = mCameraServer.putVideo("mHVSThresholdOutput", 320, 240);
        mRectanglesOutput = mCameraServer.putVideo("mRectanglesOutput", 320, 240);
    }

    public void outputColorFilters() {
        Mat thresholdOutput = mPipeline.hsvThresholdOutput();
        mHVSThresholdOutput.putFrame(thresholdOutput);
    }

    public void outputRectangles(Mat mat) {
        mRectanglesOutput.putFrame(mat);
    }

    /**
     * Called when the pipeline has run. We need to grab the output from the
     * pipeline then communicate to the rest of the system over network tables
     */
    @Override
    public void copyPipelineOutputs(CloseUpPipeline pipeline) {
        synchronized (visionLock) {

            mCallCounter++;
            // Manually makes sure that the program doesn't trip on itself
            // Take a snapshot of the pipeline's output because it may have changed the next
            // time this method is called!
            if (mCallCounter - 1 == mCounter) {
                // Pipeline work and Camera Server processing
                mPipeline = pipeline;
                // hsv output
                outputColorFilters();
                // rectangles output
                ArrayList<MatOfPoint> mats = mPipeline.findContoursOutput();
                ArrayList<RotatedRect> rectangles = new ArrayList<RotatedRect>();
                // Network Tables Stuff
                mTbl.getEntry("PIKey").setString("Call: " + mCounter);
                if (mats.size() == 2) {
                    for (int i = 0; i < 2; i++) {
                        RotatedRect rect = Imgproc.minAreaRect(new MatOfPoint2f(mats.get(i).toArray()));

                        Point center = rect.center;
                        double angle = rect.angle;
                        Size size = rect.size;
                        mTbl.getEntry("Rect " + i + " center X").setString("x = " + center.x);
                        mTbl.getEntry("Rect " + i + " center Y").setString("y = " + center.y);
                        mTbl.getEntry("Rect " + i + " angle").setDouble(angle);
                        mTbl.getEntry("Rect " + i + " Size").setString("Size : " + (size.width * size.height));


                        rectangles.add(rect);
                    }
                    //Calculate the center of mass of the tape
                    double mass1 = rectangles.get(0).size.height * rectangles.get(0).size.width;
                    double mass2 = rectangles.get(1).size.height * rectangles.get(1).size.width;
                    double massCenterXpx = ((mass1 * rectangles.get(0).center.x + mass2 * rectangles.get(1).center.x)
                            / (mass1 + mass2));
                    double massCenterYpx = ((mass1 * rectangles.get(0).center.y + mass2 * rectangles.get(1).center.y)
                            / (mass1 + mass2));

                    mTbl.getEntry("Center Of Mass X").setDouble(massCenterXpx);
                    mTbl.getEntry("Center Of Mass Y").setDouble(massCenterYpx);
                    //Calculate the distance from the target
                    double tapeDistFromCenterPxX = abs(rectangles.get(0).center.x - massCenterXpx);
                    double halfOfCameraPixelWidthInches = (TAPE_DIST_FROM_CENTER_INCHES_X / tapeDistFromCenterPxX) * HALF_OF_CAMERA_PIXEL_WIDTH_PIXELS;
                    double distanceFromTargetX = halfOfCameraPixelWidthInches / (java.lang.Math.tan(HALF_OF_CAMERA_FOV_ANGLE_X));
                    
                    mTbl.getEntry("halfOfCameraPixelWidthInches").setDouble(halfOfCameraPixelWidthInches);
                    mTbl.getEntry("Distance From Target X").setDouble(distanceFromTargetX);
                    /*
                    double tapeDistFromCenterPx = abs(rectangles.get(0).center.x - massCenterXpx);
                    double halfOfCameraPixelHeightInches = (TAPE_DIST_FROM_CENTER_INCHES_X / tapeDistFromCenterPx) * HALF_OF_CAMERA_PIXEL_WIDTH_PIXELS;
                    double distanceFromTarget = halfOfCameraPixelWidthInches / (java.lang.Math.tan(HALF_OF_CAMERA_FOV_ANGLE_X));
                    
                    mTbl.getEntry("Distance From Target X").setDouble(distanceFromTarget);*/
                }

                mCounter++;
            } else {
                mCallCounter--;
            }
        }
    }

}