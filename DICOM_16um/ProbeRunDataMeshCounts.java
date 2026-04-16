import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.util.Arrays;

public class ProbeRunDataMeshCounts {
  private static boolean hasTag(String[] tags, String needle) {
    if (tags == null) return false;
    for (String t : tags) if (needle.equals(t)) return true;
    return false;
  }

  private static int countEntities(Model m, String comp, String geom, int dim, String selTag) {
    try {
      try {
        m.component(comp).selection().remove(selTag);
      } catch (Exception ignored) {
      }
      m.component(comp).selection().create(selTag, "Explicit");
      m.component(comp).selection(selTag).geom(geom, dim);
      m.component(comp).selection(selTag).all();
      int[] e = m.component(comp).selection(selTag).entities();
      return e == null ? 0 : e.length;
    } catch (Exception ignored) {
      return -1;
    }
  }

  private static int meshElemComp(Model m, String comp, String mesh, String type) {
    try {
      return m.component(comp).mesh(mesh).getNumElem(type);
    } catch (Exception ignored) {
      return -1;
    }
  }

  private static int meshElemGlobal(Model m, String mesh, String type) {
    try {
      return m.mesh(mesh).getNumElem(type);
    } catch (Exception ignored) {
      return -1;
    }
  }

  private static void printElemBlock(String label, int v, int e, int f, int t) {
    System.out.println(label + "|V=" + v + "|E=" + e + "|F=" + f + "|T=" + t);
  }

  public static void main(String[] args) throws Exception {
    String mph = "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";
    if (args != null && args.length > 0 && args[0] != null && !args[0].isEmpty()) {
      mph = args[0];
    }
    Model m;
    try {
      m = ModelUtil.load("Model", mph);
    } catch (IOException ex) {
      throw new RuntimeException("Failed to load model: " + mph, ex);
    }

    System.out.println("MODEL|" + mph);
    try {
      System.out.println("COMPONENTS|" + Arrays.toString(m.component().tags()));
    } catch (Exception ignored) {
    }
    try {
      System.out.println("GLOBAL_MESHES|" + Arrays.toString(m.mesh().tags()));
    } catch (Exception ignored) {
    }

    if (hasTag(m.component().tags(), "comp1")) {
      try {
        System.out.println("COMP1_MESHES|" + Arrays.toString(m.component("comp1").mesh().tags()));
      } catch (Exception ignored) {
      }
      if (hasTag(m.component("comp1").mesh().tags(), "mesh2")) {
        printElemBlock(
            "COMP1_MESH2",
            meshElemComp(m, "comp1", "mesh2", "vtx"),
            meshElemComp(m, "comp1", "mesh2", "edg"),
            meshElemComp(m, "comp1", "mesh2", "tri"),
            meshElemComp(m, "comp1", "mesh2", "tet"));
      }
      if (hasTag(m.component("comp1").mesh().tags(), "mesh1")) {
        printElemBlock(
            "COMP1_MESH1",
            meshElemComp(m, "comp1", "mesh1", "vtx"),
            meshElemComp(m, "comp1", "mesh1", "edg"),
            meshElemComp(m, "comp1", "mesh1", "tri"),
            meshElemComp(m, "comp1", "mesh1", "tet"));
      }
      int b1 = countEntities(m, "comp1", "geom1", 2, "tmp_bnd_probe");
      int d1 = countEntities(m, "comp1", "geom1", 3, "tmp_dom_probe");
      System.out.println("COMP1_GEOM1|boundary_sides=" + b1 + "|domains=" + d1);
    }

    if (hasTag(m.component().tags(), "comp2")) {
      try {
        System.out.println("COMP2_MESHES|" + Arrays.toString(m.component("comp2").mesh().tags()));
      } catch (Exception ignored) {
      }
      if (hasTag(m.component("comp2").mesh().tags(), "mesh3")) {
        printElemBlock(
            "COMP2_MESH3",
            meshElemComp(m, "comp2", "mesh3", "vtx"),
            meshElemComp(m, "comp2", "mesh3", "edg"),
            meshElemComp(m, "comp2", "mesh3", "tri"),
            meshElemComp(m, "comp2", "mesh3", "tet"));
      }
      int b2 = countEntities(m, "comp2", "geom2", 2, "tmp_bnd_probe2");
      int d2 = countEntities(m, "comp2", "geom2", 3, "tmp_dom_probe2");
      System.out.println("COMP2_GEOM2|boundary_sides=" + b2 + "|domains=" + d2);
    }

    if (hasTag(m.mesh().tags(), "mpart2")) {
      printElemBlock(
          "GLOBAL_MPART2",
          meshElemGlobal(m, "mpart2", "vtx"),
          meshElemGlobal(m, "mpart2", "edg"),
          meshElemGlobal(m, "mpart2", "tri"),
          meshElemGlobal(m, "mpart2", "tet"));
    }
    if (hasTag(m.mesh().tags(), "mesh2")) {
      printElemBlock(
          "GLOBAL_MESH2",
          meshElemGlobal(m, "mesh2", "vtx"),
          meshElemGlobal(m, "mesh2", "edg"),
          meshElemGlobal(m, "mesh2", "tri"),
          meshElemGlobal(m, "mesh2", "tet"));
    }

    try {
      System.out.println("STUDIES|" + Arrays.toString(m.study().tags()));
      for (String st : m.study().tags()) {
        try {
          String[] map = m.study(st).feature("stat").getStringArray("mesh");
          System.out.println("STUDY_MESH|" + st + "|" + Arrays.toString(map));
        } catch (Exception ignored) {
        }
      }
    } catch (Exception ignored) {
    }
  }
}
