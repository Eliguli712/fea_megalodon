import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class RunStrict3BdfOvernightComp2FullRes {
  private static final String MPH = "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";

  private static final String RAW_BDF_UNCOMPRESSED =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_uncompressed_mesh3_tet_vol.bdf";

  private static final String[] STUDIES =
      new String[] {"std1", "std_nh", "std_og", "std_mr2", "std_mr5", "std_pr"};
  private static final String[] CONTACT_FEATURES = new String[] {"dcnt1", "dgcnt1"};

  private static String ts() {
    return LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
  }

  private static boolean hasTag(String[] tags, String needle) {
    if (tags == null) return false;
    for (String t : tags) {
      if (needle.equals(t)) return true;
    }
    return false;
  }

  private static String safeMsg(Throwable t) {
    if (t == null) return "";
    String m = t.getMessage();
    if (m == null || m.isEmpty()) return t.getClass().getSimpleName();
    return m.replace('\n', ' ').replace('\r', ' ');
  }

  private static void safeSetCompView(Model model, String compTag, String viewTag, String key, String value) {
    try {
      model.component(compTag).view(viewTag).set(key, value);
    } catch (Exception ignored) {
    }
  }

  private static void ensureCompView(
      Model model, String compTag, String viewTag, String label, List<String> logs) {
    try {
      if (!hasTag(model.component(compTag).view().tags(), viewTag)) {
        model.component(compTag).view().create(viewTag, 3);
      }
      try {
        model.component(compTag).view(viewTag).label(label);
      } catch (Exception ignored) {
      }
      safeSetCompView(model, compTag, viewTag, "locked", "off");
      safeSetCompView(model, compTag, viewTag, "showgrid", "off");
      safeSetCompView(model, compTag, viewTag, "showaxis", "off");
      safeSetCompView(model, compTag, viewTag, "projection", "perspective");
      safeSetCompView(model, compTag, viewTag, "rendermesh", "on");
      safeSetCompView(model, compTag, viewTag, "transparency", "off");
      logs.add("VIEW_SETUP|" + compTag + "/" + viewTag + "|ok=true");
    } catch (Exception e) {
      logs.add("VIEW_SETUP|" + compTag + "/" + viewTag + "|ok=false|err=" + safeMsg(e));
    }
  }

  private static void configureComponentMeshViews(Model model, List<String> logs) {
    if (hasTag(model.component().tags(), "comp1")) {
      ensureCompView(model, "comp1", "view_mesh1_hr", "comp1 mesh1 high-res view", logs);
      ensureCompView(model, "comp1", "view_mesh2_hr", "comp1 mesh2 high-res view", logs);
    } else {
      logs.add("VIEW_SETUP|comp1|ok=false|reason=missing_component");
    }
    if (hasTag(model.component().tags(), "comp2")) {
      ensureCompView(model, "comp2", "view_mesh3_hr", "comp2 mesh3 high-res view", logs);
    } else {
      logs.add("VIEW_SETUP|comp2|ok=false|reason=missing_component");
    }
  }

  private static int countAllEntities(Model model, int dim, String tag) {
    try {
      try {
        model.component("comp1").selection().remove(tag);
      } catch (Exception ignored) {
      }
      model.component("comp1").selection().create(tag, "Explicit");
      model.component("comp1").selection(tag).geom("geom1", dim);
      model.component("comp1").selection(tag).all();
      int[] e = model.component("comp1").selection(tag).entities();
      return e == null ? 0 : e.length;
    } catch (Exception ignored) {
      return -1;
    }
  }

  private static double evalMaxVolume(Model model, String dset, String expr, String tag) {
    try {
      try {
        model.result().numerical().remove(tag);
      } catch (Exception ignored) {
      }
      model.result().numerical().create(tag, "MaxVolume");
      model.result().numerical(tag).set("data", dset);
      model.result().numerical(tag).set("expr", new String[] {expr});
      model.result().numerical(tag).selection().all();
      model.result().numerical(tag).setResult();
      double[][] r = model.result().numerical(tag).getReal();
      if (r != null && r.length > 0 && r[0].length > 0) {
        return r[0][0];
      }
    } catch (Exception ignored) {
    }
    return Double.NaN;
  }

  private static void configureMeshFromPart(Model model, String compTag, String meshTag, String partTag, List<String> logs) {
    if (!hasTag(model.component(compTag).mesh().tags(), meshTag)) {
      throw new RuntimeException(compTag + "/" + meshTag + " is missing.");
    }
    if (!hasTag(model.component(compTag).mesh(meshTag).feature().tags(), "impmsh")) {
      model.component(compTag).mesh(meshTag).feature().create("impmsh", "Import");
    }
    model.component(compTag).mesh(meshTag).feature("impmsh").set("source", "sequence");
    model.component(compTag).mesh(meshTag).feature("impmsh").set("sequence", partTag);
    try {
      model.component(compTag).mesh(meshTag).feature("impmsh").set("buildsource", "on");
    } catch (Exception ignored) {
    }
    try {
      model.component(compTag).mesh(meshTag).feature("impmsh").set("domelemsequence", "on");
    } catch (Exception ignored) {
    }
    try {
      model.component(compTag).mesh(meshTag).feature("impmsh").set("unmesheddom", "on");
    } catch (Exception ignored) {
    }
    model.component(compTag).mesh(meshTag).run("impmsh");
    logs.add("MESH_RUN|" + compTag + "/" + meshTag + "/impmsh|ok=true|source=" + partTag);
  }

  private static int runComp1Mesh2(Model model, List<String> logs) {
    if (!hasTag(model.component("comp1").mesh().tags(), "mesh2")) {
      throw new RuntimeException("comp1/mesh2 is missing.");
    }
    try {
      String[] ftags = model.component("comp1").mesh("mesh2").feature().tags();
      if (ftags != null) {
        for (String t : ftags) {
          if (!"impmsh".equals(t) && !"fin".equals(t)) {
            try {
              model.component("comp1").mesh("mesh2").feature().remove(t);
            } catch (Exception ignored) {
            }
          }
        }
      }
    } catch (Exception ignored) {
    }

    configureMeshFromPart(model, "comp1", "mesh2", "mpart2", logs);

    int tet = -1;
    try {
      tet = model.component("comp1").mesh("mesh2").getNumElem("tet");
    } catch (Exception ignored) {
    }
    logs.add("COMP1_MESH2_TET|" + tet);
    return tet;
  }

  private static int configureRawUncompressedImport(Model model, List<String> logs) {
    if (!hasTag(model.mesh().tags(), "mpart2")) {
      throw new RuntimeException("Global mesh part mpart2 is missing.");
    }
    if (!hasTag(model.mesh("mpart2").feature().tags(), "imp1")) {
      model.mesh("mpart2").feature().create("imp1", "Import");
    }

    try {
      model.mesh("mpart2").feature("imp1").set("source", "nastran");
      model.mesh("mpart2").feature("imp1").set("filename", RAW_BDF_UNCOMPRESSED);
      model.mesh("mpart2").feature("imp1").set("createdom", "on");
      model.mesh("mpart2").feature("imp1").set("facepartition", "minimal");
      model.mesh("mpart2").run("imp1");
      logs.add("MESH_RUN|mpart2/imp1|ok=true|bdf=" + RAW_BDF_UNCOMPRESSED);
    } catch (Exception e) {
      logs.add(
          "MESH_RUN|mpart2/imp1|ok=false|bdf=" + RAW_BDF_UNCOMPRESSED + "|err=" + safeMsg(e));
      throw new RuntimeException(
          "Strict uncompressed import failed for " + RAW_BDF_UNCOMPRESSED + ": " + safeMsg(e));
    }

    try {
      configureMeshFromPart(model, "comp1", "mesh1", "mpart2", logs);
    } catch (Exception e) {
      logs.add(
          "MESH_RUN|comp1/mesh1/impmsh|ok=false|bdf="
              + RAW_BDF_UNCOMPRESSED
              + "|err="
              + safeMsg(e));
      throw new RuntimeException("Failed to bind comp1/mesh1 from strict uncompressed mpart2.");
    }

    int tet = -1;
    try {
      tet = runComp1Mesh2(model, logs);
    } catch (Exception e) {
      logs.add("MESH_RUN|comp1/mesh2|ok=false|bdf=" + RAW_BDF_UNCOMPRESSED + "|err=" + safeMsg(e));
      throw new RuntimeException("Failed to bind comp1/mesh2 from strict uncompressed mpart2.");
    }

    logs.add("RAW_BDF_IN_USE|" + RAW_BDF_UNCOMPRESSED);
    logs.add("COMP1_MESH2_TET_AFTER_STRICT_IMPORT|" + tet);
    return tet;
  }

  private static int ensureComp2Manual(Model model, List<String> logs) {
    if (hasTag(model.component().tags(), "comp2")) {
      try {
        model.component().remove("comp2");
        logs.add("COMP2_REMOVE|ok=true");
      } catch (Exception e) {
        logs.add("COMP2_REMOVE|ok=false|err=" + safeMsg(e));
      }
    }

    model.component().create("comp2");
    logs.add("COMP2_CREATE|ok=true");
    try {
      model.component("comp2").label("Component 2");
    } catch (Exception ignored) {
    }

    if (!hasTag(model.component("comp2").geom().tags(), "geom2")) {
      model.component("comp2").geom().create("geom2", 3);
      logs.add("COMP2_GEOM|geom2|ok=true");
    }

    if (!hasTag(model.component("comp2").mesh().tags(), "mesh3")) {
      model.component("comp2").mesh().create("mesh3", "geom2");
      logs.add("COMP2_MESH|mesh3|ok=true");
    }

    if (!hasTag(model.component("comp2").mesh("mesh3").feature().tags(), "impmsh")) {
      model.component("comp2").mesh("mesh3").feature().create("impmsh", "Import");
    }
    model.component("comp2").mesh("mesh3").feature("impmsh").set("source", "sequence");
    model.component("comp2").mesh("mesh3").feature("impmsh").set("sequence", "mpart2");
    try {
      model.component("comp2").mesh("mesh3").feature("impmsh").set("buildsource", "on");
    } catch (Exception ignored) {
    }
    try {
      model.component("comp2").mesh("mesh3").feature("impmsh").set("domelemsequence", "on");
    } catch (Exception ignored) {
    }
    model.component("comp2").mesh("mesh3").run("impmsh");
    logs.add("COMP2_MESH_RUN|mesh3/impmsh|ok=true|source=mpart2");

    try {
      model.component("comp2").geometricModel("mesh/mesh3");
      logs.add("COMP2_GEOMETRIC_MODEL|mesh/mesh3");
    } catch (Exception e) {
      logs.add("COMP2_GEOMETRIC_MODEL|ERR|" + safeMsg(e));
    }

    int tet = -1;
    try {
      tet = model.component("comp2").mesh("mesh3").getNumElem("tet");
    } catch (Exception e) {
      logs.add("COMP2_MESH3_TET|ERR|" + safeMsg(e));
    }
    logs.add("COMP2_MESH3_TET|" + tet);
    return tet;
  }

  private static void bindStudiesComp2First(Model model, List<String> logs) {
    for (String std : STUDIES) {
      if (!hasTag(model.study().tags(), std)) {
        logs.add("STUDY_SKIP|" + std + "|reason=missing");
        continue;
      }
      try {
        try {
          model.study(std).feature("stat").set("mesh", new String[][] {{"geom2", "mesh3"}, {"geom1", "mesh2"}});
        } catch (Exception ignored) {
          model.study(std).feature("stat").set("mesh", new String[] {"geom2", "mesh3", "geom1", "mesh2"});
        }
        logs.add("STUDY_BIND|" + std + "|mesh=geom2/mesh3,geom1/mesh2|ok=true");
      } catch (Exception e) {
        try {
          model.study(std).feature("stat").set("mesh", new String[][] {{"geom1", "mesh2"}});
          logs.add("STUDY_BIND|" + std + "|mesh=geom1/mesh2|ok=true|fallback=true");
        } catch (Exception e2) {
          logs.add("STUDY_BIND|" + std + "|mesh=geom2/mesh3,geom1/mesh2|ok=false|err=" + safeMsg(e2));
        }
      }
      try {
        model.study(std).feature("stat").set("plot", "off");
      } catch (Exception ignored) {
      }
    }
  }

  private static void safeActivateSolidFeature(Model model, String featureTag, boolean active) {
    try {
      model.component("comp1").physics("solid").feature(featureTag).active(active);
    } catch (Exception ignored) {
    }
  }

  private static void configureContactForStudy(Model model, String studyTag, List<String> logs) {
    String step = studyTag + "/stat";
    String[] solidTags;
    try {
      solidTags = model.component("comp1").physics("solid").feature().tags();
    } catch (Exception e) {
      logs.add("CONTACT_CFG|study=" + studyTag + "|ok=false|err=" + safeMsg(e));
      return;
    }

    for (String ctag : CONTACT_FEATURES) {
      if (!hasTag(solidTags, ctag)) {
        logs.add("CONTACT_CFG|study=" + studyTag + "|feature=" + ctag + "|ok=false|reason=missing");
        continue;
      }
      try {
        safeActivateSolidFeature(model, ctag, true);
        model.component("comp1").physics("solid").feature(ctag).set("StudyStep", step);
        model.component("comp1").physics("solid").feature(ctag).set("pairSelection", "all");
        logs.add("CONTACT_CFG|study=" + studyTag + "|feature=" + ctag + "|ok=true|step=" + step);
      } catch (Exception e) {
        logs.add("CONTACT_CFG|study=" + studyTag + "|feature=" + ctag + "|ok=false|err=" + safeMsg(e));
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    List<String> logs = new ArrayList<String>();
    logs.add("MODEL|" + MPH);
    logs.add("RAW_DATA_COMPRESSION|off");
    logs.add("CHUNK_POLICY|sequential_six_studies_with_gc");
    logs.add("RAW_BDF_STRICT|" + RAW_BDF_UNCOMPRESSED);
    logs.add("SOLVE_PRIORITY|comp2/mesh3_first");
    logs.add("COMPONENTS_BEFORE|" + Arrays.toString(model.component().tags()));

    int b0 = countAllEntities(model, 2, "tmp_bnd_before_overnight");
    int d0 = countAllEntities(model, 3, "tmp_dom_before_overnight");
    logs.add("ENTITY_COUNTS_BEFORE|bnd=" + b0 + "|dom=" + d0);

    int tetComp1 = configureRawUncompressedImport(model, logs);
    int tetComp2 = ensureComp2Manual(model, logs);
    configureComponentMeshViews(model, logs);
    int activeTet = tetComp2 > 0 ? tetComp2 : tetComp1;
    logs.add("ACTIVE_TET_COUNT|comp2_mesh3=" + tetComp2 + "|comp1_mesh2=" + tetComp1 + "|selected=" + activeTet);
    if (activeTet <= 0) {
      throw new RuntimeException("Strict run aborted: comp2/mesh3 and comp1/mesh2 both have nonpositive tet count.");
    }
    bindStudiesComp2First(model, logs);
    configureContactForStudy(model, "std1", logs);

    int b1 = countAllEntities(model, 2, "tmp_bnd_after_overnight");
    int d1 = countAllEntities(model, 3, "tmp_dom_after_overnight");
    logs.add("ENTITY_COUNTS_AFTER|bnd=" + b1 + "|dom=" + d1);

    String preBackup = MPH + ".bak-" + ts();
    model.save(preBackup);
    logs.add("CHECKPOINT_BACKUP|" + preBackup);
    model.save(MPH);
    logs.add("CHECKPOINT_SAVED|" + MPH);

    Map<String, String> dset = new LinkedHashMap<String, String>();
    dset.put("std1", "dset6");
    dset.put("std_nh", "dset1");
    dset.put("std_og", "dset2");
    dset.put("std_mr2", "dset3");
    dset.put("std_mr5", "dset4");
    dset.put("std_pr", "dset5");

    Map<String, String> solTag = new LinkedHashMap<String, String>();
    solTag.put("std1", "sol6");
    solTag.put("std_nh", "sol1");
    solTag.put("std_og", "sol2");
    solTag.put("std_mr2", "sol3");
    solTag.put("std_mr5", "sol4");
    solTag.put("std_pr", "sol5");

    int good = 0;
    for (String std : STUDIES) {
      if (!hasTag(model.study().tags(), std)) {
        continue;
      }

      String stSol = solTag.get(std);
      configureContactForStudy(model, std, logs);
      if (stSol != null && hasTag(model.sol().tags(), stSol)) {
        try {
          model.sol(stSol).clearSolutionData();
        } catch (Exception ignored) {
        }
        try {
          model.sol(stSol).clearSolution();
        } catch (Exception ignored) {
        }
        logs.add("SOL_CLEAR|" + stSol + "|study=" + std + "|ok=true");
      }

      try {
        model.study(std).run();
        logs.add("STUDY_RUN|" + std + "|ok=true");
      } catch (Exception e) {
        logs.add("STUDY_RUN|" + std + "|ok=false|err=" + safeMsg(e));
        continue;
      }

      String ds = dset.get(std);
      double vm = evalMaxVolume(model, ds, "solid.mises", "mxvm_overnight_" + std);
      double um = evalMaxVolume(model, ds, "sqrt(u^2+v^2+w^2)", "mxu_overnight_" + std);
      boolean ok =
          Double.isFinite(vm)
              && Double.isFinite(um)
              && Math.abs(vm) > 1e-12
              && Math.abs(um) > 1e-15
              && activeTet > 0;
      if (ok) {
        good++;
      }
      logs.add(
          "CHECK|study="
              + std
              + "|dataset="
              + ds
              + "|vm="
              + vm
              + "|um="
              + um
              + "|finite_nonzero="
              + ok);

      try {
        System.gc();
      } catch (Exception ignored) {
      }

      try {
        model.save(MPH);
        logs.add("INTERMEDIATE_SAVE|study=" + std + "|ok=true");
      } catch (Exception e) {
        logs.add("INTERMEDIATE_SAVE|study=" + std + "|ok=false|err=" + safeMsg(e));
      }
    }

    String backup = MPH + ".bak-" + ts();
    model.save(backup);
    logs.add("BACKUP|" + backup);
    model.save(MPH);
    logs.add("SAVED|" + MPH);

    logs.add("SUMMARY|finite_nonzero_studies=" + good + "|total_target_studies=" + STUDIES.length);
    logs.add("COMPONENTS_AFTER|" + Arrays.toString(model.component().tags()));

    for (String line : logs) {
      System.out.println(line);
    }
  }
}
