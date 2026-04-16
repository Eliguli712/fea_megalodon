import com.comsol.model.*;
import com.comsol.model.util.*;

import java.util.Arrays;

public class TestManualComp2Build {
  private static boolean has(String[] arr, String k) {
    if (arr == null) return false;
    for (String s : arr) if (k.equals(s)) return true;
    return false;
  }

  public static void main(String[] args) throws Exception {
    String mph = "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf_comp2trial.mph";
    Model m = ModelUtil.load("Model", mph);

    System.out.println("COMP_BEFORE|" + Arrays.toString(m.component().tags()));

    if (!has(m.component().tags(), "comp2")) {
      m.component().create("comp2");
      System.out.println("COMP2_CREATE|ok=true");
    }

    try { m.component("comp2").label("Component 2"); } catch (Exception ignored) {}

    if (!has(m.component("comp2").geom().tags(), "geom1")) {
      m.component("comp2").geom().create("geom1", 3);
      System.out.println("COMP2_GEOM_CREATE|geom1|ok=true");
    }

    if (!has(m.component("comp2").mesh().tags(), "mesh2")) {
      m.component("comp2").mesh().create("mesh2", "geom1");
      System.out.println("COMP2_MESH_CREATE|mesh2|ok=true");
    }

    if (!has(m.component("comp2").mesh("mesh2").feature().tags(), "impmsh")) {
      m.component("comp2").mesh("mesh2").create("impmsh", "Import");
    }
    m.component("comp2").mesh("mesh2").feature("impmsh").set("source", "sequence");
    m.component("comp2").mesh("mesh2").feature("impmsh").set("sequence", "mpart2");
    try { m.component("comp2").mesh("mesh2").feature("impmsh").set("buildsource", "on"); } catch (Exception ignored) {}
    try { m.component("comp2").mesh("mesh2").feature("impmsh").set("domelemsequence", "on"); } catch (Exception ignored) {}
    m.component("comp2").mesh("mesh2").run("impmsh");

    try { m.component("comp2").geometricModel("mesh/mesh2"); } catch (Exception e) { System.out.println("COMP2_GEOM_MODEL|ERR|" + e.getMessage()); }

    int tet = -1;
    try { tet = m.component("comp2").mesh("mesh2").getNumElem("tet"); } catch (Exception e) { System.out.println("COMP2_TET|ERR|" + e.getMessage()); }
    System.out.println("COMP2_TET|" + tet);

    System.out.println("COMP_AFTER|" + Arrays.toString(m.component().tags()));
    System.out.println("COMP2_GEOMS|" + Arrays.toString(m.component("comp2").geom().tags()));
    System.out.println("COMP2_MESHES|" + Arrays.toString(m.component("comp2").mesh().tags()));

    m.save(mph);
    System.out.println("SAVED|" + mph);
  }
}
