import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class RunLivytanVolumetricStudies {
  private static final String BDF =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth.bdf";
  private static final String OUT_MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth_volsolve.mph";

  private static void p(String s) {
    System.out.println(s);
  }

  private static void safeSetSolid(Model m, String feat, String key, String value) {
    try {
      m.component("comp1").physics("solid").feature(feat).set(key, value);
    } catch (Exception ignored) {}
  }

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
    try {
      m.study(studyTag).feature("stat").set("geometricNonlinearity", "off");
    } catch (Exception ignored) {}
  }

  public static Model run() {
    p("BDF_FILE|" + BDF);

    Model m = ModelUtil.create("Model");
    m.modelNode().create("mod1");
    m.component().create("comp1", false);
    m.component("comp1").geom().create("geom1", 3);
    m.component("comp1").mesh().create("mesh1");
    m.component("comp1").mesh("mesh1").feature().create("imp1", "Import");
    m.component("comp1").mesh("mesh1").feature("imp1").set("source", "nastran");
    m.component("comp1").mesh("mesh1").feature("imp1").set("filename", BDF);

    m.component("comp1").mesh("mesh1").run("imp1");
    try {
      m.component("comp1").mesh("mesh1").run("fin");
    } catch (Exception ignored) {}
    p("Imported volumetric BDF mesh.");

    m.component("comp1").physics().create("solid", "SolidMechanics", "geom1");
    m.component("comp1").physics("solid").selection().all();
    // Ensure linear elastic material is fully defined for solver assembly.
    safeSetSolid(m, "lemm1", "E_mat", "userdef");
    safeSetSolid(m, "lemm1", "E", "1.5e8[Pa]");
    safeSetSolid(m, "lemm1", "nu_mat", "userdef");
    safeSetSolid(m, "lemm1", "nu", "0.30");
    safeSetSolid(m, "lemm1", "rho_mat", "userdef");
    safeSetSolid(m, "lemm1", "rho", "1100[kg/m^3]");

    m.component("comp1").physics("solid").create("fix1", "Fixed", 2);
    m.component("comp1").physics("solid").feature("fix1").selection().geom("geom1", 2);
    m.component("comp1").physics("solid").feature("fix1").selection().all();

    m.component("comp1").physics("solid").create("body1", "BodyLoad", 3);
    m.component("comp1").physics("solid").feature("body1").selection().geom("geom1", 3);
    m.component("comp1").physics("solid").feature("body1").selection().all();

    int domCount = m.component("comp1").physics("solid").selection().entities(3).length;
    int bndCount = m.component("comp1").physics("solid").feature("fix1").selection().entities(2).length;
    p("SELECTION_COUNTS|domains=" + domCount + "|fixed_boundaries=" + bndCount);
    if (domCount <= 0) {
      throw new RuntimeException("Imported mesh has 0 selected solid domains.");
    }

    // Study 1
    m.param().set("body_load_y", "5e4[N/m^3]");
    m.component("comp1").physics("solid").feature("body1").set("FperVol", new String[]{"0", "-body_load_y", "0"});
    ensureStudy(m, "std1", "Livytan volumetric solid study 1");
    m.study("std1").run();
    p("STUDY_DONE|std1");

    // Study 2 with a different load level
    m.param().set("body_load_y", "8e4[N/m^3]");
    m.component("comp1").physics("solid").feature("body1").set("FperVol", new String[]{"0", "-body_load_y", "0"});
    ensureStudy(m, "std2", "Livytan volumetric solid study 2");
    m.study("std2").run();
    p("STUDY_DONE|std2");

    try {
      m.save(OUT_MPH);
      p("SAVED|" + OUT_MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to save model: " + e.getMessage(), e);
    }

    return m;
  }

  public static void main(String[] args) {
    run();
  }
}
