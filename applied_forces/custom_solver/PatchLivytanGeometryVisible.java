import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class PatchLivytanGeometryVisible {
  private static final String MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth_volsolve.mph";
  private static final String BDF =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth.bdf";
  private static final String STL =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth.stl";

  private static void p(String s) { System.out.println(s); }

  public static void main(String[] args) {
    Model m;
    try {
      m = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model", e);
    }

    // Add an explicit geometry feature so the Geometry tree is not empty in GUI.
    try { m.component("comp1").geom("geom1").feature().remove("impviz"); } catch (Exception ignored) {}
    m.component("comp1").geom("geom1").feature().create("impviz", "Import");
    m.component("comp1").geom("geom1").feature("impviz").set("filename", STL);
    try { m.component("comp1").geom("geom1").feature("impviz").set("facepartition", "minimal"); } catch (Exception ignored) {}
    try { m.component("comp1").geom("geom1").feature("impviz").set("selresult", "on"); } catch (Exception ignored) {}
    m.component("comp1").geom("geom1").run();

    // Reassert full-res volumetric mesh import.
    try { m.component("comp1").mesh("mesh1").feature().remove("imp1"); } catch (Exception ignored) {}
    m.component("comp1").mesh("mesh1").feature().create("imp1", "Import");
    m.component("comp1").mesh("mesh1").feature("imp1").set("source", "nastran");
    m.component("comp1").mesh("mesh1").feature("imp1").set("filename", BDF);
    m.component("comp1").mesh("mesh1").run("imp1");
    try { m.component("comp1").mesh("mesh1").run("fin"); } catch (Exception ignored) {}

    // Keep BC/domain selections non-empty.
    try { m.component("comp1").physics("solid").selection().all(); } catch (Exception ignored) {}
    try {
      m.component("comp1").physics("solid").feature("fix1").selection().geom("geom1", 2);
      m.component("comp1").physics("solid").feature("fix1").selection().all();
    } catch (Exception ignored) {}
    try {
      m.component("comp1").physics("solid").feature("body1").selection().geom("geom1", 3);
      m.component("comp1").physics("solid").feature("body1").selection().all();
    } catch (Exception ignored) {}

    p("GEOM_FEATS|" + String.join(",", m.component("comp1").geom("geom1").feature().tags()));
    p("GEOM_COUNTS|dom=" + m.component("comp1").geom("geom1").getNDomains()
        + "|bnd=" + m.component("comp1").geom("geom1").getNBoundaries()
        + "|edg=" + m.component("comp1").geom("geom1").getNEdges());
    p("MESH_COUNTS|v=" + m.component("comp1").mesh("mesh1").getNumVertex()
        + "|tri=" + m.component("comp1").mesh("mesh1").getNumElem("tri")
        + "|tet=" + m.component("comp1").mesh("mesh1").getNumElem("tet"));
    try {
      int nd = m.component("comp1").physics("solid").selection().entities(3).length;
      int nb = m.component("comp1").physics("solid").feature("fix1").selection().entities(2).length;
      p("SELECTION_COUNTS|domains=" + nd + "|fixed_boundaries=" + nb);
    } catch (Exception e) {
      p("SELECTION_COUNTS|error|" + e.getMessage());
    }

    try {
      m.save(MPH);
      p("SAVED|" + MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to save model", e);
    }
  }
}
