import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;

public class EmbedVonMisesImagesToStaticDynamics {
  private static final String MODEL_PATH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics.mph";

  private static final String EXPORTS =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports";

  private static final String[][] IMAGES = new String[][]{
      {"smoothed", "surface_mesh_smoothed"},
      {"uncompressed", "tooth_surface_uncompressed"},
      {"rawtet", "tooth_surface_comsol_tet_vol"},
  };

  private static void upsertImagePlot(Model m, String pgTag, String label, String imagePath) {
    try {
      m.result().remove(pgTag);
    } catch (Exception ignored) {
    }
    m.result().create(pgTag, "PlotGroup2D");
    m.result(pgTag).label(label);
    try {
      m.result(pgTag).set("data", "none");
    } catch (Exception ignored) {
    }
    m.result(pgTag).create("img1", "Image");
    ResultFeature img = m.result(pgTag).feature("img1");
    img.set("sourcetype", "user");
    img.set("filename", imagePath);
    try {
      img.set("mapping", "auto");
      img.set("heightmode", "fit");
      img.set("coordinterpretation", "pixels");
      img.set("planetype", "xy");
      img.set("preserveaspect", true);
    } catch (Exception ignored) {
    }
    m.result(pgTag).run();
    System.out.println("IMAGE_EMBEDDED|" + pgTag + "|" + imagePath);
  }

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MODEL_PATH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MODEL_PATH, e);
    }

    for (String[] item : IMAGES) {
      String shortTag = item[0];
      String name = item[1];
      String pointPng = EXPORTS + "/von_mises_point_cloud_" + shortTag + ".png";
      String surfacePng = EXPORTS + "/von_mises_surface_" + shortTag + ".png";

      upsertImagePlot(
          model,
          "pg_vm_point_img_" + shortTag,
          "Von Mises Point Cloud Image - " + name,
          pointPng
      );
      upsertImagePlot(
          model,
          "pg_vm_surface_img_" + shortTag,
          "Von Mises Surface Image - " + name,
          surfacePng
      );
    }

    try {
      model.save(MODEL_PATH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to save model: " + MODEL_PATH, e);
    }
    System.out.println("Saved: " + MODEL_PATH);
  }
}
