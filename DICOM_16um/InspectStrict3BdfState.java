import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.util.Arrays;

public class InspectStrict3BdfState {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution.mph";

  private static String safeGet(PropFeature f, String key) {
    try {
      String v = f.getString(key);
      return v == null ? "" : v;
    } catch (Exception ignored) {
      return "";
    }
  }

  private static void printMeshInfo(Model model, String owner, String meshTag, MeshFeatureList fl) {
    try {
      System.out.println(owner + "_MESH_FEATURE_TAGS|mesh=" + meshTag + "|tags=" + Arrays.toString(fl.tags()));
      for (String ft : fl.tags()) {
        PropFeature f = (PropFeature) fl.get(ft);
        String type;
        try {
          type = f.getType();
        } catch (Exception ignored) {
          type = "";
        }
        String src = safeGet(f, "source");
        String seq = safeGet(f, "sequence");
        String fn = safeGet(f, "filename");
        if (!src.isEmpty() || !seq.isEmpty() || !fn.isEmpty() || "Import".equals(type)) {
          System.out.println(
              owner
                  + "_MESH_FEATURE|mesh="
                  + meshTag
                  + "|tag="
                  + ft
                  + "|type="
                  + type
                  + "|source="
                  + src
                  + "|sequence="
                  + seq
                  + "|filename="
                  + fn);
        }
      }
    } catch (Exception e) {
      System.out.println(owner + "_MESH_INFO_ERROR|mesh=" + meshTag + "|err=" + e.getMessage());
    }
  }

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    System.out.println("MODEL|" + MPH);
    System.out.println("GLOBAL_MESH_TAGS|" + Arrays.toString(model.mesh().tags()));
    for (String gTag : model.mesh().tags()) {
      printMeshInfo(model, "GLOBAL", gTag, model.mesh(gTag).feature());
    }
    try {
      System.out.println(
          "MPART1_FEATURE_TAGS|" + Arrays.toString(model.mesh("mpart1").feature().tags()));
      PropFeature imp1 = model.mesh("mpart1").feature("imp1");
      System.out.println("MPART1_IMP1_SOURCE|" + safeGet(imp1, "source"));
      System.out.println("MPART1_IMP1_FILE|" + safeGet(imp1, "filename"));
      System.out.println("MPART1_IMP1_CREATEDOM|" + safeGet(imp1, "createdom"));
      System.out.println("MPART1_IMP1_FACEPARTITION|" + safeGet(imp1, "facepartition"));
    } catch (Exception e) {
      System.out.println("MPART1_INFO_ERROR|" + e.getMessage());
    }

    try {
      String[] compMeshTags = model.component("comp1").mesh().tags();
      System.out.println("COMP1_MESH_TAGS|" + Arrays.toString(compMeshTags));
      for (String cTag : compMeshTags) {
        printMeshInfo(model, "COMP1", cTag, model.component("comp1").mesh(cTag).feature());
      }
    } catch (Exception e) {
      System.out.println("COMP1_MESH_INFO_ERROR|" + e.getMessage());
    }

    System.out.println("STUDY_TAGS|" + Arrays.toString(model.study().tags()));
    for (String std : model.study().tags()) {
      try {
        String[] stf = model.study(std).feature().tags();
        System.out.println("STUDY_FEATURES|tag=" + std + "|features=" + Arrays.toString(stf));
        for (String ft : stf) {
          try {
            String[] meshMap = model.study(std).feature(ft).getStringArray("mesh");
            if (meshMap != null && meshMap.length > 0) {
              System.out.println("STUDY_MESH|study=" + std + "|feature=" + ft + "|mesh=" + Arrays.toString(meshMap));
            }
          } catch (Exception ignored) {
          }
        }
      } catch (Exception e) {
        System.out.println("STUDY_ERROR|tag=" + std + "|err=" + e.getMessage());
      }
    }
  }
}
