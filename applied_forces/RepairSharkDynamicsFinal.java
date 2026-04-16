import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class RepairSharkDynamicsFinal {
  private static int[] keptDomains() {
    int[] rem = new int[]{2,5,6,25,28,46,48,51,62,84,91,98,100,106,110,116,121,127,131,135,152,165,182};
    boolean[] keep = new boolean[184];
    for (int i=1;i<=183;i++) keep[i] = true;
    for (int r: rem) keep[r] = false;
    int n=0; for (int i=1;i<=183;i++) if (keep[i]) n++;
    int[] out = new int[n]; int k=0;
    for (int i=1;i<=183;i++) if (keep[i]) out[k++]=i;
    return out;
  }

  private static void p(String s) { System.out.println(s); }

  private static double evalMaxSelected(Model m, String tag, String dset, int[] dom) {
    try { m.result().numerical().remove(tag); } catch (Exception e) {}
    try {
      m.result().numerical().create(tag, "MaxVolume");
      m.result().numerical(tag).set("expr", new String[]{"solid.mises"});
      m.result().numerical(tag).set("unit", new String[]{"Pa"});
      m.result().numerical(tag).set("data", dset);
      m.result().numerical(tag).selection().geom("geom1",3);
      m.result().numerical(tag).selection().set(dom);
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      return (r!=null&&r.length>0&&r[0].length>0)?r[0][0]:Double.NaN;
    } catch (Exception e) {
      p("eval " + tag + " failed: " + e.getMessage());
      return Double.NaN;
    }
  }

  private static void configureCommonPhysics(Model m, int[] dom) {
    m.component("comp1").physics("solid").selection().geom("geom1",3);
    m.component("comp1").physics("solid").selection().set(dom);

    // Linear material fallback parameters on the active meshed domains.
    try { m.component("comp1").physics("solid").feature("lemm1").selection().geom("geom1",3); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").selection().set(dom); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("E_mat","userdef"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("E","1.5e8[Pa]"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("nu_mat","userdef"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("nu","0.3"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("rho_mat","userdef"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("rho","1100[kg/m^3]"); } catch (Exception e) {}

    // Snout-localized domain force (stable vs boundary singularities on fragmented boundaries).
    m.param().set("thrust_vol", "2e4[N/m^3]");
    m.param().descr("thrust_vol", "Body-force equivalent thrust applied in snout region.");

    try { m.component("comp1").selection().remove("sel_snout_dom"); } catch (Exception e) {}
    m.component("comp1").selection().create("sel_snout_dom", "Box");
    m.component("comp1").selection("sel_snout_dom").set("entitydim", "3");
    m.component("comp1").selection("sel_snout_dom").set("xmin", "-1e9[m]");
    m.component("comp1").selection("sel_snout_dom").set("xmax", "1e9[m]");
    m.component("comp1").selection("sel_snout_dom").set("ymin", "-1e9[m]");
    m.component("comp1").selection("sel_snout_dom").set("ymax", "1e9[m]");
    m.component("comp1").selection("sel_snout_dom").set("zmin", "20[m]");
    m.component("comp1").selection("sel_snout_dom").set("zmax", "1e9[m]");

    try { m.component("comp1").physics("solid").feature().remove("body1"); } catch (Exception e) {}
    m.component("comp1").physics("solid").create("body1", "BodyLoad", 3);
    m.component("comp1").physics("solid").feature("body1").selection().named("sel_snout_dom");
    m.component("comp1").physics("solid").feature("body1").set("FperVol", new String[]{"0","0","thrust_vol"});
    m.component("comp1").physics("solid").feature("body1").active(true);

    // Robust constraints: fully fix external boundaries to remove rigid/void/singular modes.
    try { m.component("comp1").physics("solid").feature("fix1"); }
    catch (Exception e) { m.component("comp1").physics("solid").create("fix1", "Fixed", 2); }
    m.component("comp1").physics("solid").feature("fix1").selection().geom("geom1",2);
    m.component("comp1").physics("solid").feature("fix1").selection().all();
    m.component("comp1").physics("solid").feature("fix1").active(true);

    try { m.component("comp1").physics("solid").feature("rms1").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("bndl1").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("bndl_pr").active(false); } catch (Exception e) {}

    // Hyperelastic parameters.
    m.param().set("kappa_bulk", "2.5e8[Pa]");
    m.param().set("mu_ref", "2.5e7[Pa]");
    m.param().set("ogden_mu1", "2.2e7[Pa]");
    m.param().set("ogden_alpha1", "1.3");

    try { m.component("comp1").physics("solid").feature("hmm_nh").set("MaterialModel", "NeoHookean"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_nh").set("G_mat", "userdef"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_nh").set("G", "mu_ref"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_nh").set("K_mat", "userdef"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_nh").set("K", "kappa_bulk"); } catch (Exception e) {}

    try { m.component("comp1").physics("solid").feature("hmm_og").set("MaterialModel", "Ogden"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_og").set("mup", "ogden_mu1"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_og").set("alphap", "ogden_alpha1"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_og").set("kappa", "kappa_bulk"); } catch (Exception e) {}
  }

  private static void configureStudyMesh(Model m, String st) {
    try { m.study(st).feature("stat").set("mesh", new String[][]{{"geom1","mesh2"}}); } catch (Exception e) {}
    try { m.study(st).feature("stat").set("geometricNonlinearity", "off"); } catch (Exception e) {}
    try { m.study(st).feature("stat").set("plot", "off"); } catch (Exception e) {}
  }

  private static void activateLinear(Model m) {
    try { m.component("comp1").physics("solid").feature("lemm1").active(true); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_nh").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_og").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_mr2").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_mr5").active(false); } catch (Exception e) {}
  }

  private static void activateNH(Model m) {
    try { m.component("comp1").physics("solid").feature("lemm1").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_nh").active(true); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_og").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_mr2").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_mr5").active(false); } catch (Exception e) {}
  }

  private static void activateOG(Model m) {
    try { m.component("comp1").physics("solid").feature("lemm1").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_nh").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_og").active(true); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_mr2").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_mr5").active(false); } catch (Exception e) {}
  }

  private static void ensurePlot(Model m, String pg, String dset, String label) {
    try { m.result().remove(pg); } catch (Exception e) {}
    m.result().create(pg, "PlotGroup3D");
    m.result(pg).label(label);
    m.result(pg).set("data", dset);
    m.result(pg).create("surf1", "Surface");
    m.result(pg).feature("surf1").set("expr", "solid.mises");
    m.result(pg).feature("surf1").set("unit", "Pa");
    m.result(pg).feature("surf1").set("descr", "Von Mises stress");
    m.result(pg).run();
  }

  public static void main(String[] args) {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    String out = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/RepairSharkDynamicsFinal_Model.mph";

    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    int[] dom = keptDomains();

    // Remove malformed torus feature that causes repeated syntax errors.
    try { m.geom("part1").feature().remove("tor1"); p("Removed part1/tor1"); } catch (Exception e) { p("tor1 removal skipped"); }

    configureCommonPhysics(m, dom);
    configureStudyMesh(m, "std1");
    configureStudyMesh(m, "std_nh");
    configureStudyMesh(m, "std_og");
    configureStudyMesh(m, "std_mr2");
    configureStudyMesh(m, "std_mr5");
    configureStudyMesh(m, "std_pr");

    // 1) Linear stable reference
    activateLinear(m);
    try { m.study("std1").run(); p("std1 complete"); }
    catch (Exception e) { p("std1 failed: " + e.getMessage()); }
    double mxStd1 = evalMaxSelected(m, "mx_std1_sel", "dset6", dom);
    p("std1 max mises (selected)=" + mxStd1);

    // 2) Neo-Hookean
    activateNH(m);
    try { m.study("std_nh").run(); p("std_nh complete"); }
    catch (Exception e) { p("std_nh failed: " + e.getMessage()); }
    double mxNh = evalMaxSelected(m, "mx_nh_sel", "dset1", dom);
    p("std_nh max mises (selected)=" + mxNh);

    // 3) Ogden
    activateOG(m);
    try { m.study("std_og").run(); p("std_og complete"); }
    catch (Exception e) { p("std_og failed: " + e.getMessage()); }
    double mxOg = evalMaxSelected(m, "mx_og_sel", "dset2", dom);
    p("std_og max mises (selected)=" + mxOg);

    // Restore Neo-Hookean as default active hyperelastic model.
    activateNH(m);

    // Stable, non-empty stress cloud plots bound to solved datasets.
    try { ensurePlot(m, "pg_vms_std1", "dset6", "Von Mises Stress Cloud (std1)"); } catch (Exception e) { p("plot std1 failed: " + e.getMessage()); }
    try { ensurePlot(m, "pg_vms_nh", "dset1", "Von Mises Stress Cloud (Neo-Hookean)"); } catch (Exception e) { p("plot nh failed: " + e.getMessage()); }
    try { ensurePlot(m, "pg_vms_og", "dset2", "Von Mises Stress Cloud (Ogden)"); } catch (Exception e) { p("plot og failed: " + e.getMessage()); }

    try { m.save(out); }
    catch (IOException e) { throw new RuntimeException(e); }
  }
}
