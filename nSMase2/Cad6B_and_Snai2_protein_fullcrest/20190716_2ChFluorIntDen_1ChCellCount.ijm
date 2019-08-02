//learn file name, prepare file and Fiji for analysis
name=File.nameWithoutExtension;
rename("A");
run("Rotate... ", "angle=180 grid=1 interpolation=Bilinear stack");
run("Z Project...", "projection=[Max Intensity]");
setTool("freehand");
run("Set Measurements...", "area mean min integrated display redirect=None decimal=3");

//Close unnecessary windows from last analysis
if (isOpen("Results")) { 
         selectWindow("Results"); 
         run("Close"); 
    } 
if (isOpen("Summary")) { 
         selectWindow("Summary"); 
         run("Close"); 
    } 
if (isOpen("ROI Manager")) { 
         selectWindow("ROI Manager"); 
         run("Close"); 
    } 

//Optional: Input ROI File:
//roi=File.openDialog("Select ROI file");
//roiManager("Open",roi);

//Define ROIs Manually (background, experimental and control sides)
waitForUser("Draw Background ROI 1, then press ok");
roiManager("Add");
roiManager("Select",0);
roiManager("Rename","background");
roiManager("Show All");
waitForUser("Draw Background ROI 2, then press ok");
roiManager("Add");
roiManager("Select",1);
roiManager("Rename","background");
waitForUser("Draw Background ROI 3, then press ok");
roiManager("Add");
roiManager("Select",2);
roiManager("Rename","background");
waitForUser("Draw Experimental ROI, then press ok");
roiManager("Add");
roiManager("Select",3);
roiManager("Rename","Expt");
waitForUser("Draw Control ROI, then press ok");
roiManager("Add");
roiManager("Select",4);
roiManager("Rename","Cntl");
run("Split Channels");

//Measure background then ROI IntDen
//Channel 1
selectWindow("C1-MAX_A");
rename("Cad6B");
resetMinAndMax();
roiManager("Show All");
roiManager("Select", 0);
run("Measure");
roiManager("Select", 1);
run("Measure");
roiManager("Select", 2);
run("Measure");
roiManager("Select", 4);
run("Measure");
roiManager("Select", 3);
run("Measure");

//Channel 2
selectWindow("C2-MAX_A");
rename("Snai2");
resetMinAndMax();
roiManager("Show All");
roiManager("Select", 0);
run("Measure");
roiManager("Select", 1);
run("Measure");
roiManager("Select", 2);
run("Measure");
roiManager("Select", 4);
run("Measure");
roiManager("Select", 3);
run("Measure");

//Save out ROIs
waitForUser("Choose a directory to save ROIs");
dir = getDirectory("Choose a directory to save ROI sets");
roiManager("Save", dir+name+".zip");

//Save out Measurements as csv
waitForUser("Choose a directory to save IntDen measurements");
dir = getDirectory("Choose a directory to save CSV measurement results");
saveAs("Results", dir+name+".csv");

//Analyze cell counts
selectWindow("Snai2");
run("Median...", "radius=2 slice");
resetMinAndMax();
run("8-bit");
rename("RAW");
run("Auto Local Threshold", "method=Bernsen radius=15 parameter_1=0 parameter_2=0 white");
roiManager("Show All");
run("Analyze Particles...", "size=5.00-Infinity show=Masks");
run("Invert LUT");
rename("CntlSide");
run("Duplicate...", "title=ExptSide");

selectWindow("CntlSide");
roiManager("Show All");
roiManager("Select", 4);
run("Analyze Particles...", "size=5-Infinity show=Nothing summarize");

selectWindow("ExptSide");
roiManager("Show All");
roiManager("Select", 3);
run("Analyze Particles...", "size=5-Infinity show=Nothing summarize");

//Save out CSVs
waitForUser("Choose a directory to save Cell Counts, then press ok");
dir = getDirectory("Choose a directory to save measurement results.");
saveAs("Results", dir+name+".csv");


//Close image windows
run("Close All");