"""A1-size build-plate recovery fixture, not a calibrated A1 digital twin.

Released parts use rigid contact dynamics. The stuck case uses a permanent weld,
not a fitted adhesion or fracture law. The controller never observes that flag.
No printer connections, temperature simulation, learned policy or camera detector.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).parent
VENDOR = ROOT / 'vendor/franka_emika_panda'
DT = .04
BED_TOP = .75
BED_HALF = .128
COOL_LIMIT_C = 35.
CONTACT_STOP_N = 2.0  # Synthetic monitoring threshold, not an A1 damage limit.
DOWN = np.diag([1., -1., -1.])
HOME = np.array([0., 0., 0., -1.57079, 0., 1.57079, -.7853])


@dataclass(frozen=True)
class Scenario:
    name: str
    half_size: tuple = (.020, .020, .015)
    position_xy: tuple = (-.045, .025)
    yaw_deg: float = 0.
    mass_kg: float = .025
    friction: float = .3
    stuck: bool = False
    bed_temperature_c: float = 25.
    residue: bool = False


CASES = (
    Scenario('released_small'),
    Scenario('released_wide', half_size=(.040, .035, .008), position_xy=(-.025, -.035), yaw_deg=12., mass_kg=.035),
    Scenario('stuck_stop', stuck=True),
)


def numbers(values):
    return ' '.join(str(float(v)) for v in values)


def add_camera(world, name, pos, look, fovy):
    pos, look = np.array(pos), np.array(look)
    z = pos - look
    z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(world, 'camera', name=name, pos=numbers(pos), xyaxes=numbers(np.r_[x, y]), fovy=str(fovy))


def scene_xml(case):
    root = ET.parse(VENDOR / 'panda.xml').getroot()
    root.set('model', 'a1_size_recovery_fixture')
    root.find('compiler').set('meshdir', str(VENDOR / 'assets'))
    root.find('option').set('timestep', '.002')
    root.find('default').insert(0, ET.Element('geom', solref='.008 1', solimp='.95 .99 .001', friction='.3 .005 .0001'))
    key = root.find('keyframe')
    if key is not None:
        root.remove(key)
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='640')
    assets = root.find('asset')
    ET.SubElement(assets, 'texture', name='background', type='skybox', builtin='gradient',
                  rgb1='.8 .83 .86', rgb2='.96 .96 .96', width='128', height='128')
    world = root.find('worldbody')
    arm = world.find("body[@name='link0']")
    arm.set('pos', '-.50 0 .68')
    hand = arm.find(".//body[@name='hand']")
    tool = ET.SubElement(hand, 'body', name='recovery_paddle')
    ET.SubElement(tool, 'geom', name='paddle_stem', type='capsule', fromto='0 0 .10 0 0 .15',
                  size='.008', mass='.02', rgba='.25 .28 .3 1')
    ET.SubElement(tool, 'geom', name='pusher', type='box', pos='0 0 .16', size='.008 .025 .009',
                  mass='.03', priority='1', friction='.3 .005 .0001', rgba='.18 .20 .23 1')
    ET.SubElement(tool, 'site', name='tool_tip', pos='0 0 .16', size='.002', rgba='0 0 0 0')
    ET.SubElement(world, 'geom', name='ground', type='plane', size='2 2 .1', rgba='.88 .88 .88 1')
    ET.SubElement(world, 'geom', name='pedestal', type='box', pos='-.50 0 .34', size='.09 .09 .34', rgba='.3 .32 .35 1')
    ET.SubElement(world, 'geom', name='bed_support', type='box', pos='0 0 .70', size='.11 .11 .043', rgba='.20 .22 .24 1')
    ET.SubElement(world, 'geom', name='build_plate', type='box', pos='0 0 .746', size='.128 .128 .004',
                  friction=numbers([case.friction, .005, .0001]), rgba='.66 .53 .26 1')
    # The geometry below is an invented collection fixture, not A1 hardware CAD.
    ET.SubElement(world, 'geom', name='bin_floor', type='box', pos='.29 0 .602', size='.16 .18 .008', rgba='.50 .55 .60 1')
    for name, pos, size in (
        ('bin_front', (.445, 0, .66), (.005, .18, .05)),
        ('bin_back', (.135, 0, .64), (.005, .18, .03)),
        ('bin_left', (.29, -.175, .66), (.15, .005, .05)),
        ('bin_right', (.29, .175, .66), (.15, .005, .05)),
    ):
        ET.SubElement(world, 'geom', name=name, type='box', pos=numbers(pos), size=numbers(size), rgba='.50 .55 .60 1')
    angle = np.deg2rad(case.yaw_deg)
    part = ET.SubElement(world, 'body', name='printed_part',
                         pos=numbers([*case.position_xy, BED_TOP + case.half_size[2] + .0001]),
                         quat=numbers([np.cos(angle / 2), 0, 0, np.sin(angle / 2)]))
    ET.SubElement(part, 'freejoint', name='part_joint')
    ET.SubElement(part, 'geom', name='part', type='box', size=numbers(case.half_size),
                  mass=str(case.mass_kg), friction=numbers([case.friction, .005, .0001]), rgba='.85 .35 .10 1')
    if case.stuck:
        ET.SubElement(root.find('equality'), 'weld', name='synthetic_attachment', body1='printed_part', solref='.002 1')
    if case.residue:
        residue = ET.SubElement(world, 'body', name='residue', pos='-.075 -.09 .751')
        ET.SubElement(residue, 'geom', name='residue', type='box', size='.015 .005 .001', rgba='.85 .35 .10 1')
    ET.SubElement(world, 'camera', name='inspection', pos='0 0 1.35', xyaxes='1 0 0 0 1 0', fovy='33')
    add_camera(world, 'overview', [.80, -1.05, 1.48], [-.12, 0, .85], 44)
    add_camera(world, 'detail', [.35, -.47, 1.25], [.10, 0, .70], 42)
    ET.SubElement(world, 'light', pos='0 -.6 1.8', dir='0 0 -1', diffuse='.6 .6 .6')
    return ET.tostring(root, encoding='unicode')


class RecoveryExperiment:
    def __init__(self, case, render=False, detector=None):
        self.case = case
        self.detector = detector
        self.inspection = None
        self.inspection_renderer = None
        self.model = mujoco.MjModel.from_xml_string(scene_xml(case))
        self.data = mujoco.MjData(self.model)
        self.ik = mujoco.MjData(self.model)
        self.joints = np.array([self.model.joint(f'joint{i}').id for i in range(1, 8)])
        self.qadr, self.dadr = self.model.jnt_qposadr[self.joints], self.model.jnt_dofadr[self.joints]
        self.limits = self.model.jnt_range[self.joints]
        self.site = self.model.site('tool_tip').id
        self.part = self.model.body('printed_part').id
        self.part_geom = self.model.geom('part').id
        self.pusher = self.model.geom('pusher').id
        self.part_dof = self.model.joint('part_joint').dofadr[0]
        self.target = np.array([-.17, 0., .89])
        self.qtarget = HOME.copy()
        self.qtarget = self.solve_ik(self.target)
        self.data.qpos[self.qadr] = self.qtarget
        self.data.ctrl[:7] = self.qtarget
        self.data.ctrl[7] = .015 / .04 * 255
        for name in ('finger_joint1', 'finger_joint2'):
            self.data.qpos[self.model.joint(name).qposadr[0]] = .015
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, 480, 640) if render else None
        self.records, self.frames = [], []
        self.phase = 'Initialize'
        self.status = 'not_started'
        self.force_stop = False
        self.guard_enabled = False
        self.peak_force = 0.
        self.contact_steps = 0
        self.unexpected_contacts = set()
        self.min_joint_margin = float('inf')
        self.parked = True
        self.last_command = np.zeros(3)
        self.hold(.4, 'Initialize')
        self.initial_position = self.position().copy()

    def solve_ik(self, target):
        q = self.qtarget.copy()
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(220):
            self.ik.qpos[:] = self.data.qpos
            self.ik.qpos[self.qadr] = q
            mujoco.mj_forward(self.model, self.ik)
            rotation = self.ik.site_xmat[self.site].reshape(3, 3)
            ep = target - self.ik.site_xpos[self.site]
            er = .5 * sum(np.cross(rotation[:, i], DOWN[:, i]) for i in range(3))
            if np.linalg.norm(ep) < .00005 and np.linalg.norm(er) < .0002 and np.trace(rotation.T @ DOWN) > 2.99:
                return q
            mujoco.mj_jacSite(self.model, self.ik, jp, jr, self.site)
            j = np.vstack([jp[:, self.dadr], jr[:, self.dadr]])
            dq = j.T @ np.linalg.solve(j @ j.T + .002**2 * np.eye(6), np.r_[ep, er])
            dq *= min(1., .12 / max(np.max(np.abs(dq)), 1e-10))
            q = np.clip(q + dq, self.limits[:, 0] + .01, self.limits[:, 1] - .01)
        raise RuntimeError(f'Unreachable tool target {target}; IK error {np.linalg.norm(ep):.4f} m')

    def position(self):
        return self.data.xpos[self.part].copy()

    def speed(self):
        return float(np.linalg.norm(self.data.qvel[self.part_dof:self.part_dof + 3]))

    def corners(self, geom):
        half = self.model.geom_size[geom]
        local = np.array([[x, y, z] for x in (-half[0], half[0])
                          for y in (-half[1], half[1]) for z in (-half[2], half[2])])
        return local @ self.data.geom_xmat[geom].reshape(3, 3).T + self.data.geom_xpos[geom]

    def bed_clear(self):
        geoms = [self.part_geom]
        if self.case.residue:
            geoms.append(self.model.geom('residue').id)
        for gid in geoms:
            c = self.corners(gid)
            low, high = c.min(axis=0), c.max(axis=0)
            if low[0] < BED_HALF and high[0] > -BED_HALF and low[1] < BED_HALF and high[1] > -BED_HALF and high[2] > BED_TOP - .005:
                return False
        return True

    def inspect_plate(self):
        if self.detector is None:
            return self.bed_clear()
        if self.inspection is None:
            if self.inspection_renderer is None:
                self.inspection_renderer = mujoco.Renderer(self.model, 128, 128)
            self.inspection_renderer.update_scene(self.data, camera='inspection')
            rgb = self.inspection_renderer.render().copy()
            self.inspection = self.detector.predict(rgb)
            self.inspection['image'] = rgb
        return bool(self.inspection['clear'])

    def collected(self):
        c = self.corners(self.part_geom)
        return bool(np.all(c[:, 0] > .14) and np.all(c[:, 0] < .44)
                    and np.all(c[:, 1] > -.17) and np.all(c[:, 1] < .17)
                    and np.min(c[:, 2]) >= .605 and np.max(c[:, 2]) < .73 and self.speed() < .01)

    def step(self, velocity, phase):
        self.phase = phase
        self.last_command = np.array(velocity, dtype=float)
        self.target += self.last_command * DT
        self.qtarget = self.solve_ik(self.target)
        self.data.ctrl[:7] = self.qtarget
        force = np.zeros(6)
        step_force, contacts = 0., 0
        for _ in range(round(DT / self.model.opt.timestep)):
            self.data.qfrc_applied[self.dadr] = self.data.qfrc_bias[self.dadr]
            mujoco.mj_step(self.model, self.data)
            total_force, touching = 0., False
            for i, c in enumerate(self.data.contact):
                pair = {int(c.geom1), int(c.geom2)}
                if pair == {self.part_geom, self.pusher}:
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    total_force += float(np.linalg.norm(force[:3]))
                    touching = True
                # Robot against fixture surfaces, excluding its mounting pedestal.
                for gid in pair:
                    if self.model.geom_bodyid[gid] != 0:
                        continue
                    name = self.model.geom(gid).name
                    other = next(iter(pair - {gid}), gid)
                    bid = self.model.geom_bodyid[other]
                    bodyname = self.model.body(bid).name
                    if bodyname not in ('world', 'printed_part', 'residue', 'link0') and name != 'pedestal':
                        self.unexpected_contacts.add(name)
            contacts += touching
            self.contact_steps += touching
            step_force = max(step_force, total_force)
            if self.guard_enabled and total_force > CONTACT_STOP_N and not self.force_stop:
                self.force_stop = True
                # Freeze the joint target immediately; never reset physical position/velocity.
                self.qtarget = self.data.qpos[self.qadr].copy()
                self.data.ctrl[:7] = self.qtarget
                self.target = self.data.site_xpos[self.site].copy()
            q = self.data.qpos[self.qadr]
            self.min_joint_margin = min(self.min_joint_margin, float(np.min(np.minimum(q - self.limits[:, 0], self.limits[:, 1] - q))))
        self.peak_force = max(self.peak_force, step_force)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError('Non-finite simulation state')
        self.records.append(dict(t=float(self.data.time), phase=phase, position=self.position().tolist(),
                                 speed=self.speed(), force_n=step_force, contact_fraction=contacts / round(DT / self.model.opt.timestep),
                                 tool_position=self.data.site_xpos[self.site].tolist(), target=self.target.tolist(),
                                 velocity_command=self.last_command.tolist(), joint_positions=self.data.qpos[self.qadr].tolist(),
                                 bed_clear=self.bed_clear(), collected=self.collected(), force_stop=self.force_stop))
        if self.renderer:
            pair = []
            for cam in ('overview', 'detail'):
                self.renderer.update_scene(self.data, camera=cam)
                pair.append(self.renderer.render().copy())
            self.frames.append(np.concatenate(pair, axis=1))

    def hold(self, seconds, phase):
        for _ in range(round(seconds / DT)):
            self.step([0, 0, 0], phase)

    def move(self, destination, phase, speed=.08):
        destination = np.array(destination)
        for _ in range(300):
            error = destination - self.target
            distance = np.linalg.norm(error)
            if distance < .0001:
                break
            velocity = error / max(distance, 1e-10) * min(speed, distance / DT)
            self.step(velocity, phase)
            if self.force_stop and self.guard_enabled:
                break
        else:
            raise RuntimeError('Cartesian motion time limit')
        if not self.force_stop:
            self.hold(.20, phase)

    def run(self):
        if self.case.bed_temperature_c is None or not np.isfinite(self.case.bed_temperature_c):
            self.status = 'temperature_unknown'
            self.hold(.6, 'Temperature unknown')
            return self.report()
        if self.case.bed_temperature_c > COOL_LIMIT_C:
            self.status = 'waiting_for_cooldown'
            self.hold(.6, 'Wait for cooldown')
            return self.report()
        self.parked = False
        corners = self.corners(self.part_geom)
        rear = float(corners[:, 0].min()) - .008 - .004
        y = float(self.position()[1])
        self.move([rear, y, .89], 'Approach')
        self.move([rear, y, .761], 'Lower', speed=.08)
        self.guard_enabled = True
        self.status = 'pushing'
        start = self.data.time
        previous = self.position().copy()
        no_progress = 0.
        for _ in range(350):
            self.step([.035, 0, 0], 'Recover part')
            if self.force_stop:
                self.status = 'attached_or_obstructed'
                break
            if self.position()[0] - previous[0] > .001:
                previous = self.position().copy()
                no_progress = 0.
            else:
                no_progress += DT
            if no_progress > 1. and self.contact_steps > 0:
                self.status = 'no_progress'
                break
            if self.bed_clear() or self.position()[2] < .72:
                self.status = 'part_left_plate'
                break
            if self.target[0] > .205:
                self.status = 'travel_limit'
                break
        else:
            self.status = 'timeout'
        self.push_time = self.data.time - start
        self.guard_enabled = False
        # One withdrawal after recovery or abort; no automatic retries against adhesion.
        self.move(self.target + [-.035, 0, .10], 'Withdraw', speed=.08)
        self.move([-.17, 0, .89], 'Park', speed=.10)
        self.hold(.8, 'Verify collection')
        self.parked = np.linalg.norm(self.data.site_xpos[self.site] - np.array([-.17, 0, .89])) < .005
        if self.status == 'part_left_plate':
            self.status = 'recovered' if self.collected() and self.inspect_plate() else 'collection_or_clearance_failed'
        self.hold(.4, 'Result')
        return self.report()

    def report(self):
        ready = (self.status == 'recovered' and self.inspect_plate() and self.collected() and self.parked
                 and not self.force_stop and not self.unexpected_contacts and self.min_joint_margin > 0
                 and self.case.bed_temperature_c is not None
                 and np.isfinite(self.case.bed_temperature_c) and self.case.bed_temperature_c <= COOL_LIMIT_C)
        return dict(case=asdict(self.case), status=self.status, ready_for_next_print=bool(ready),
                    part_collected=self.collected(), bed_clear=self.bed_clear(), parked=bool(self.parked),
                    initial_position_m=self.initial_position.tolist(), final_position_m=self.position().tolist(),
                    contact_steps=self.contact_steps, peak_contact_force_n=self.peak_force,
                    force_stop_triggered=self.force_stop, unintended_fixture_contacts=sorted(self.unexpected_contacts),
                    min_joint_margin_rad=self.min_joint_margin, simulation_time_s=float(self.data.time),
                    actual_printer_command_sent=False,
                    inspection_mode='learned_rgb_segmentation' if self.detector is not None else 'simulator_oracle',
                    inspection={k:v for k,v in self.inspection.items() if k not in ('image', 'mask', 'probability')} if self.inspection is not None else None)

    def close(self):
        if self.inspection_renderer:
            self.inspection_renderer.close()
        if self.renderer:
            self.renderer.close()


def trial(case, render=False):
    env = RecoveryExperiment(case, render=render)
    try:
        return env.run(), env.records, env.frames
    finally:
        env.close()


def export(output, results):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    title = ImageFont.load_default(size=26)
    font = ImageFont.load_default(size=22)
    with imageio.get_writer(output / 'recovery_demo.mp4', fps=25, codec='libx264', quality=8, macro_block_size=1) as writer:
        for report, records, frames in results:
            for i in range(len(frames) + 30):
                k = min(i, len(frames) - 1)
                r = records[k]
                canvas = Image.new('RGB', (1280, 656), '#f8f8f8')
                canvas.paste(Image.fromarray(frames[k]), (0, 104))
                draw = ImageDraw.Draw(canvas)
                draw.text((20, 12), 'A1-SIZE PLATE / PART RECOVERY PROTOTYPE', fill='black', font=title)
                draw.text((20, 47), f"{report['case']['name']}  |  {r['phase']}  |  t={r['t']:.2f}s", fill='black', font=font)
                draw.text((20, 77), 'Learned RGB inspection / Released or stuck rigid part / Not a calibrated printer model' if report.get('inspection_mode') == 'learned_rgb_segmentation' else 'Released rigid part or synthetic stuck case / No learned AI / Not a calibrated printer model', fill='black', font=font)
                draw.text((20, 595), f"Part in bin: {r['collected']}   |   Plate clear: {r['bed_clear']}   |   Contact: {r['force_n']:.2f} N", fill='black', font=font)
                if k == len(frames) - 1:
                    text = 'SIMULATED NEXT-PRINT GATE: READY' if report['ready_for_next_print'] else 'SIMULATED NEXT-PRINT GATE: BLOCKED'
                else:
                    text = 'RECOVERY STOPPED: ATTACHED OR OBSTRUCTED' if r['force_stop'] else 'No real printer is connected'
                draw.text((20, 625), text, fill='black', font=font)
                writer.append_data(np.asarray(canvas))
                if i == len(frames) - 1:
                    canvas.save(output / f"{report['case']['name']}_final.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [trial(case, args.render) for case in CASES]
    metadata = dict(stage='physical_fixture_prototype_no_learned_ai',
                    scope='Released part recovery and stuck-part stop; adhesion/fracture not simulated',
                    bed_half_size_m=BED_HALF, bed_top_m=BED_TOP, cooldown_gate_c=COOL_LIMIT_C,
                    synthetic_contact_stop_n=CONTACT_STOP_N,
                    source_manifest=json.loads((VENDOR / 'source_manifest.json').read_text()),
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    scene_sha256={case.name: hashlib.sha256(scene_xml(case).encode()).hexdigest() for case in CASES},
                    versions={name: importlib.metadata.version(name) for name in ('mujoco', 'numpy')},
                    results=[r[0] for r in results])
    (args.output / 'metrics.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (args.output / 'trajectories.json').write_text(json.dumps([r[1] for r in results]) + '\n')
    if args.render:
        export(args.output, results)
    print(json.dumps(metadata['results'], indent=2))


if __name__ == '__main__':
    main()
