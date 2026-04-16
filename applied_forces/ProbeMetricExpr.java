import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeMetricExpr {
  static String tx = "(solid.sx*nx + solid.sxy*ny + solid.sxz*nz)";
  static String ty = "(solid.sxy*nx + solid.sy*ny + solid.syz*nz)";
  static String tz = "(solid.sxz*nx + solid.syz*ny + solid.sz*nz)";
  static String tmag = "sqrt(("+tx+")^2 + ("+ty+")^2 + ("+tz+")^2)";
  static String mdens = "sqrt((y*("+tz+")-z*("+ty+"))^2 + (z*("+tx+")-x*("+tz+"))^2 + (x*("+ty+")-y*("+tx+"))^2)";

  static double eval(Model m, String tag, String type, String expr, int dim, String sel, String dset) {
    try { m.result().numerical().remove(tag); } catch (Exception e) {}
    m.result().numerical().create(tag, type);
    m.result().numerical(tag).set("expr", new String[]{expr});
    m.result().numerical(tag).set("data", dset);
    m.result().numerical(tag).selection().named(sel);
    m.result().numerical(tag).setResult();
    double[][] r = m.result().numerical(tag).getReal();
    return (r!=null && r.length>0 && r[0].length>0) ? r[0][0] : Double.NaN;
  }

  public static void main(String[] args) throws Exception {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.component("comp1").selection().remove("sel_front_probe2"); } catch (Exception e) {}
    try { m.component("comp1").selection().remove("sel_tail_probe2"); } catch (Exception e) {}
    m.component("comp1").selection().create("sel_front_probe2", "Box");
    m.component("comp1").selection("sel_front_probe2").set("entitydim", 2);
    m.component("comp1").selection("sel_front_probe2").set("xmin", -100.0);
    m.component("comp1").selection("sel_front_probe2").set("xmax", 100.0);
    m.component("comp1").selection("sel_front_probe2").set("ymin", -100.0);
    m.component("comp1").selection("sel_front_probe2").set("ymax", 100.0);
    m.component("comp1").selection("sel_front_probe2").set("zmin", 21.4);
    m.component("comp1").selection("sel_front_probe2").set("zmax", 100.0);

    m.component("comp1").selection().create("sel_tail_probe2", "Box");
    m.component("comp1").selection("sel_tail_probe2").set("entitydim", 2);
    m.component("comp1").selection("sel_tail_probe2").set("xmin", -100.0);
    m.component("comp1").selection("sel_tail_probe2").set("xmax", 100.0);
    m.component("comp1").selection("sel_tail_probe2").set("ymin", -100.0);
    m.component("comp1").selection("sel_tail_probe2").set("ymax", 100.0);
    m.component("comp1").selection("sel_tail_probe2").set("zmin", -100.0);
    m.component("comp1").selection("sel_tail_probe2").set("zmax", 10.2);

    String dset = "dset4";
    double tforce = eval(m, "int_tail_f", "IntSurface", tmag, 2, "sel_tail_probe2", dset);
    double impactNm = eval(m, "int_tail_m", "IntSurface", mdens, 2, "sel_tail_probe2", dset);
    double instant = eval(m, "max_front_w", "MaxSurface", "("+tmag+")*1[m/s]", 2, "sel_front_probe2", dset);
    double avgm = eval(m, "avg_mises", "AverageVolume", "solid.mises", 3, "geom1_dom", dset);
    double maxm = eval(m, "max_mises", "MaxVolume", "solid.mises", 3, "geom1_dom", dset);
    System.out.println("tail force N = " + tforce);
    System.out.println("impact Nm = " + impactNm);
    System.out.println("instant W/m2 = " + instant);
    System.out.println("avg mises Pa = " + avgm);
    System.out.println("max mises Pa = " + maxm);
  }
}
