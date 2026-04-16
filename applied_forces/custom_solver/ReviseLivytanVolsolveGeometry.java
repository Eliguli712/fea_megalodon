import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ReviseLivytanVolsolveGeometry {
  private static final String MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth_volsolve.mph";
  private static final String BDF =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth.bdf";
  private static final String STL =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth.stl";

  private static void p(String s) { System.out.println(s); }

  private static void ensureStudy(Model m, String studyTag, String label) {
    try {
      m.study(studyTag);
    } catch (Exception e) {
      m.study().create(studyTag);
    }
    m.study(studyTag).label(label);
    try {
      m.study(studyTag).feature("stat");
    } catch (Exception e) {
      m.study(studyTag).create("stat", "Stationary");
    }
    m.study(studyTag).feature("stat").activate("solid", true);
    try {
      m.study(studyTag).feature("stat").set("mesh", new String[][]{{"geom1", "mesh1"}});
    } catch (Exception ignored) {}
    try {
      m.study(studyTag).feature("stat").set("plot", "off");
    } catch (Exception ignored) {}
  }

  public static void main(String[] args) {
    Model m;
    try {
      m = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load MPH", e);
    }

    // Make geometry explicitly visible in the builder by adding an STL import feature.
    try { m.component("comp1").geom("geom1").feature().remove("impviz"); } catch (Exception ignored) {}
    m.component("comp1").geom("geom1").feature().create("impviz", "Import");
    m.component("comp1").geom("geom1").feature("impviz").set("filename", STL);
    try { m.component("comp1").geom("geom1").feature("impviz").set("facepartition", "minimal"); } catch (Exception ignored) {}
    try { m.component("comp1").geom("geom1").feature("impviz").set("selresult", "on"); } catch (Exception ignored) {}

    m.component("comp1").geom("geom1").run();
    p("GEOM_COUNTS|dom=" + m.component("comp1").geom("geom1").getNDomains()
        + "|bnd=" + m.component("comp1").geom("geom1").getNBoundaries()
        + "|edg=" + m.component("comp1").geom("geom1").getNEdges());

    // Rebuild mesh import from full-resolution volumetric BDF.
    try { m.component("comp1").mesh("mesh1").feature().remove("imp1"); } catch (Exception ignored) {}
    m.component("comp1").mesh("mesh1").feature().create("imp1", "Import");
    m.component("comp1").mesh("mesh1").feature("imp1").set("source", "nastran");
    m.component("comp1").mesh("mesh1").feature("imp1").set("filename", BDF);
    m.component("comp1").mesh("mesh1").run("imp1");
    try { m.component("comp1").mesh("mesh1").run("fin"); } catch (Exception ignored) {}
    p("MESH_COUNTS|v=" + m.component("comp1").mesh("mesh1").getNumVertex()
        + "|tri=" + m.component("comp1").mesh("mesh1").getNumElem("tri")
        + "|tet=" + m.component("comp1").mesh("mesh1").getNumElem("tet"));

    // Keep physics selections valid and non-empty.
    try {
      m.component("comp1").physics("solid").selection().all();
    } catch (Exception e) {
      m.component("comp1").physics().create("solid", "SolidMechanics", "geom1");
      m.component("comp1").physics("solid").selection().all();
    }

    try { m.component("comp1").physics("solid").feature("fix1"); }
    catch (Exception e) { m.component("comp1").physics("solid").create("fix1", "Fixed", 2); }
    m.component("comp1").physics("solid").feature("fix1").selection().geom("geom1", 2);
    m.component("comp1").physics("solid").feature("fix1").selection().all();

    try { m.component("comp1").physics("solid").feature("body1"); }
    catch (Exception e) { m.component("comp1").physics("solid").create("body1", "BodyLoad", 3); }
    m.component("comp1").physics("solid").feature("body1").selection().geom("geom1", 3);
    m.component("comp1").physics("solid").feature("body1").selection().all();

    int domCount = m.component("comp1").physics("solid").selection().entities(3).length;
    int bndCount = m.component("comp1").physics("solid").feature("fix1").selection().entities(2).length;
    p("SELECTION_COUNTS|domains=" + domCount + "|fixed_boundaries=" + bndCount);
    if (domCount <= 0) {
      throw new RuntimeException("No solid domains selected after geometry revision.");
    }

    // Re-run the two existing stationary studies to ensure non-zero DOF persists.
    ensureStudy(m, "std1", "Livytan volumetric solid study 1");
    m.study("std1").run();
    p("STUDY_DONE|std1");

    ensureStudy(m, "std2", "Livytan volumetric solid study 2");
    m.study("std2").run();
    p("STUDY_DONE|std2");

    try {
      m.save(MPH);
      p("SAVED|" + MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to save revised MPH", e);
    }
  }
}
