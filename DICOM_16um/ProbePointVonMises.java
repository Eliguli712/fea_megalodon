import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;

public class ProbePointVonMises {
  private static final String MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics.mph";
  private static final String PNG =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/probe_vm_point.png";

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    try {
      model.result().remove("pg_probe_vm_point");
    } catch (Exception ignored) {
    }
    model.result().create("pg_probe_vm_point", "PlotGroup3D");
    model.result("pg_probe_vm_point").label("Probe Von Mises Point");
    try {
      model.result("pg_probe_vm_point").set("data", "dset4");
    } catch (Exception ignored) {
    }
    model.result("pg_probe_vm_point").create("pt1", "Point");
    ResultFeature pt = model.result("pg_probe_vm_point").feature("pt1");
    try {
      pt.selection().geom("geom1", 2);
      pt.selection().all();
    } catch (Exception ignored) {
    }
    try {
      pt.set("expr", new String[]{"solid.mises"});
    } catch (Exception ignored) {
    }
    try {
      pt.set("descr", new String[]{"Von Mises stress"});
    } catch (Exception ignored) {
    }
    try {
      pt.set("colortable", "RainbowLight");
    } catch (Exception ignored) {
    }
    try {
      pt.set("resolution", "normal");
    } catch (Exception ignored) {
    }
    try {
      pt.set("pointsizeactive", true);
    } catch (Exception ignored) {
    }
    try {
      pt.set("pointsize", 2.0);
    } catch (Exception ignored) {
    }
    try {
      pt.set("pointshape", "sphere");
    } catch (Exception ignored) {
    }
    try {
      pt.set("smooth", "internal");
    } catch (Exception ignored) {
    }

    try {
      model.result("pg_probe_vm_point").run();
      System.out.println("POINT_PLOT_OK");
    } catch (Exception e) {
      System.out.println("POINT_PLOT_FAIL " + e.getMessage());
      throw e;
    }

    try {
      model.result().export().remove("img_probe_vm_point");
    } catch (Exception ignored) {
    }
    model.result().export().create("img_probe_vm_point", "Image");
    model.result().export("img_probe_vm_point").set("plotgroup", "pg_probe_vm_point");
    try {
      model.result().export("img_probe_vm_point").set("imagetype", "png");
    } catch (Exception ignored) {
    }
    try {
      model.result().export("img_probe_vm_point").set("size", "manual");
    } catch (Exception ignored) {
    }
    try {
      model.result().export("img_probe_vm_point").set("unit", "px");
    } catch (Exception ignored) {
    }
    try {
      model.result().export("img_probe_vm_point").set("width", 1400);
    } catch (Exception ignored) {
    }
    try {
      model.result().export("img_probe_vm_point").set("height", 1050);
    } catch (Exception ignored) {
    }
    model.result().export("img_probe_vm_point").set("pngfilename", PNG);
    model.result().export("img_probe_vm_point").run();
    System.out.println("POINT_EXPORT_OK " + PNG);
  }
}
