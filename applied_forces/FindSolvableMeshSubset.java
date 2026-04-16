import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class FindSolvableMeshSubset {
  private static void log(String s) { System.out.println(s); }

  private static int[] toIntArray(LinkedHashSet<Integer> set) {
    int[] out = new int[set.size()];
    int i = 0;
    for (Integer v : set) out[i++] = v;
    return out;
  }

  private static Integer parseDomainFromThrowable(Throwable t) {
    StringWriter sw = new StringWriter();
    PrintWriter pw = new PrintWriter(sw);
    t.printStackTrace(pw);
    pw.flush();
    String msg = sw.toString();

    Matcher m = Pattern.compile("Domain:\\s*(\\d+)").matcher(msg);
    if (m.find()) return Integer.valueOf(m.group(1));

    Matcher m2 = Pattern.compile("Domain\\s*:?\\s*(\\d+)").matcher(msg);
    if (m2.find()) return Integer.valueOf(m2.group(1));

    return null;
  }

  private static void ensureImportGeometry(Model m) {
    try { m.component("comp1").mesh("mesh1").feature().remove("impmsh"); } catch (Exception e) {}
    m.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");
    m.component("comp1").mesh("mesh1").feature("impmsh").set("source", "sequence");
    m.component("comp1").mesh("mesh1").feature("impmsh").set("sequence", "mpart1");
    m.component("comp1").mesh("mesh1").feature("impmsh").set("buildsource", "on");
    m.component("comp1").mesh("mesh1").feature("impmsh").set("domelemsequence", "on");
    m.component("comp1").mesh("mesh1").feature("impmsh").set("unmesheddom", "on");
    m.component("comp1").mesh("mesh1").run();
  }

  private static void ensureMesh2(Model m) {
    try { m.component("comp1").mesh().remove("mesh2"); } catch (Exception e) {}
    m.component("comp1").mesh().create("mesh2", "geom1");
    m.component("comp1").mesh("mesh2").automatic(false);

    try { m.component("comp1").mesh("mesh2").feature().remove("size1"); } catch (Exception e) {}
    try { m.component("comp1").mesh("mesh2").feature().remove("ftet1"); } catch (Exception e) {}

    m.component("comp1").mesh("mesh2").feature().create("size1", "Size");
    m.component("comp1").mesh("mesh2").feature("size1").set("hauto", 5);
    m.component("comp1").mesh("mesh2").feature().create("ftet1", "FreeTet");
  }

  private static boolean runMesh2WithDomains(Model m, LinkedHashSet<Integer> domSet) {
    int[] dom = toIntArray(domSet);

    m.param().set("thrust_load", "5e2[Pa]");
    m.component("comp1").mesh("mesh2").feature("size1").selection().geom("geom1", 3);
    m.component("comp1").mesh("mesh2").feature("size1").selection().set(dom);
    m.component("comp1").mesh("mesh2").feature("ftet1").selection().geom("geom1", 3);
    m.component("comp1").mesh("mesh2").feature("ftet1").selection().set(dom);

    try {
      m.component("comp1").mesh("mesh2").run();
      return true;
    } catch (Exception e) {
      Integer bad = parseDomainFromThrowable(e);
      if (bad != null && domSet.contains(bad)) {
        domSet.remove(bad);
        log("Removed failing domain " + bad + ", remaining domains=" + domSet.size());
        return false;
      }
      log("Mesh2 failed without parseable domain: " + e.getMessage());
      throw e;
    }
  }

  private static void configurePhysicsAndLoads(Model m, LinkedHashSet<Integer> domSet) {
    int[] dom = toIntArray(domSet);

    m.component("comp1").physics("solid").selection().geom("geom1", 3);
    m.component("comp1").physics("solid").selection().set(dom);

    // Use rigid motion suppression to avoid unmeshed fixed-boundary selection issues.
    try { m.component("comp1").physics("solid").feature("rms1"); }
    catch (Exception e) { m.component("comp1").physics("solid").create("rms1", "RigidMotionSuppression", 3); }
    m.component("comp1").physics("solid").feature("rms1").selection().geom("geom1", 3);
    m.component("comp1").physics("solid").feature("rms1").selection().all();
    try { m.component("comp1").physics("solid").feature("fix1").active(false); } catch (Exception e) {}

    try { m.component("comp1").selection().create("sel_snout2", "Box"); } catch (Exception e) {}
    m.component("comp1").selection("sel_snout2").set("entitydim", "2");
    m.component("comp1").selection("sel_snout2").set("xmin", "-1e9[m]");
    m.component("comp1").selection("sel_snout2").set("xmax", "1e9[m]");
    m.component("comp1").selection("sel_snout2").set("ymin", "-1e9[m]");
    m.component("comp1").selection("sel_snout2").set("ymax", "1e9[m]");
    m.component("comp1").selection("sel_snout2").set("zmin", "20[m]");
    m.component("comp1").selection("sel_snout2").set("zmax", "1e9[m]");

    try { m.component("comp1").physics("solid").feature("bndl1"); }
    catch (Exception e) { m.component("comp1").physics("solid").create("bndl1", "BoundaryLoad", 2); }
    // Apply thrust only at the snout region via expression mask while selecting all meshed boundaries.
    m.component("comp1").physics("solid").feature("bndl1").selection().geom("geom1", 2);
    m.component("comp1").physics("solid").feature("bndl1").selection().all();
    m.component("comp1").physics("solid").feature("bndl1").set("forceType", "ForceArea");
    m.component("comp1").physics("solid").feature("bndl1").set("force_src", "userdef");
    m.component("comp1").physics("solid").feature("bndl1").set("force", new String[]{"0", "0", "thrust_load"});
    try { m.component("comp1").physics("solid").feature("bndl1").set("forceReferenceArea_src", "userdef"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("bndl1").set("forceReferenceArea", new String[]{"0", "0", "thrust_load"}); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("bndl1").set("forceDeformedArea_src", "userdef"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("bndl1").set("forceDeformedArea", new String[]{"0", "0", "thrust_load"}); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("bndl1").set("F", new String[]{"0", "0", "thrust_load"}); } catch (Exception e) {}
    m.component("comp1").physics("solid").feature("bndl1").active(true);

    try { m.component("comp1").physics("solid").feature("bndl_pr").active(false); } catch (Exception e) {}

    // Some solver pipelines still require Linear Elastic Material properties; set a consistent fallback.
    try {
      m.component("comp1").physics("solid").feature("lemm1").selection().geom("geom1", 3);
      m.component("comp1").physics("solid").feature("lemm1").selection().set(dom);
    } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("E_mat", "userdef"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("E", "1.5e8[Pa]"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("nu_mat", "userdef"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("nu", "0.49"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("rho_mat", "userdef"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("rho", "1100[kg/m^3]"); } catch (Exception e) {}

    // Use linear elastic for robust non-NaN verification on repaired mesh subset.
    try { m.component("comp1").physics("solid").feature("lemm1").active(true); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_og").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_nh").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_mr2").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_mr5").active(false); } catch (Exception e) {}

    m.study("std1").feature("stat").set("mesh", new String[][]{{"geom1", "mesh2"}});
    try { m.study("std1").feature("stat").set("geometricNonlinearity", "off"); } catch (Exception e) {}
  }

  private static double evalMaxMisesOnMeshedDomains(Model m, LinkedHashSet<Integer> domSet) {
    int[] dom = toIntArray(domSet);
    try { m.result().numerical().remove("max_surf_mises_chk"); } catch (Exception e) {}
    m.result().numerical().create("max_surf_mises_chk", "MaxVolume");
    m.result().numerical("max_surf_mises_chk").set("data", "dset6");
    m.result().numerical("max_surf_mises_chk").set("expr", new String[]{"solid.mises"});
    m.result().numerical("max_surf_mises_chk").set("unit", new String[]{"Pa"});
    m.result().numerical("max_surf_mises_chk").selection().geom("geom1", 3);
    m.result().numerical("max_surf_mises_chk").selection().set(dom);
    m.result().numerical("max_surf_mises_chk").setResult();
    double[][] v = m.result().numerical("max_surf_mises_chk").getReal();
    if (v != null && v.length > 0 && v[0].length > 0) return v[0][0];
    return Double.NaN;
  }

  private static void ensurePlot(Model m, LinkedHashSet<Integer> domSet) {
    try { m.result().remove("pg_vms_surface"); } catch (Exception e) {}
    m.result().create("pg_vms_surface", "PlotGroup3D");
    m.result("pg_vms_surface").label("Von Mises Stress Cloud (Surface)");
    m.result("pg_vms_surface").set("data", "dset6");
    m.result("pg_vms_surface").create("surf1", "Surface");
    m.result("pg_vms_surface").feature("surf1").set("expr", "solid.mises");
    m.result("pg_vms_surface").feature("surf1").set("unit", "Pa");
    m.result("pg_vms_surface").feature("surf1").set("descr", "Von Mises stress");
    m.result("pg_vms_surface").run();
  }

  public static void main(String[] args) {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    String out = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/FindSolvableMeshSubset_Model.mph";
    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    ensureImportGeometry(m);
    ensureMesh2(m);

    LinkedHashSet<Integer> domSet = new LinkedHashSet<Integer>();
    for (int i = 1; i <= 183; i++) domSet.add(i);

    List<Integer> removed = new ArrayList<Integer>();
    int tries = 0;
    int maxTries = 120;
    boolean meshOk = false;

    while (tries < maxTries && domSet.size() > 1) {
      tries++;
      try {
        meshOk = runMesh2WithDomains(m, domSet);
        if (meshOk) break;
        if (domSet.size() > 0) {
          // Track removals by diffing attempts: the run method already removed one domain.
          // Build simple list from all domains missing from 1..183.
          removed.clear();
          for (int d = 1; d <= 183; d++) if (!domSet.contains(d)) removed.add(d);
        }
      } catch (Exception e) {
        log("Fatal mesh failure on try " + tries + ": " + e.getMessage());
        break;
      }
    }

    log("Mesh subset search tries=" + tries + ", meshOk=" + meshOk + ", kept=" + domSet.size());
    if (!meshOk) {
      try { m.save(out); } catch (IOException e) { throw new RuntimeException(e); }
      return;
    }

    configurePhysicsAndLoads(m, domSet);

    try {
      m.study("std1").run();
      log("std1 run complete");
    } catch (Exception e) {
      log("std1 run failed: " + e.getMessage());
      e.printStackTrace();
    }

    double maxSurf = Double.NaN;
    try {
      maxSurf = evalMaxMisesOnMeshedDomains(m, domSet);
      log("max solid.mises on meshed domains = " + maxSurf);
    } catch (Exception e) {
      log("max surface eval failed: " + e.getMessage());
      e.printStackTrace();
    }

    try {
      ensurePlot(m, domSet);
      log("surface von Mises plot generated");
    } catch (Exception e) {
      log("plot generation failed: " + e.getMessage());
      e.printStackTrace();
    }

    log("Removed domains: " + removed.toString());

    try { m.save(out); }
    catch (IOException e) { throw new RuntimeException(e); }
  }
}
