import express from 'express';
import { PartnershipService } from '../services/PartnershipService';

const router = express.Router();
const partnershipService = new PartnershipService();

router.get('/', async (req, res) => {
  const partnerships = await partnershipService.getAllPartnerships();
  res.json(partnerships);
});

router.get('/:id', async (req, res) => {
  const partnership = await partnershipService.getPartnershipById(req.params.id);
  res.json(partnership);
});

router.post('/', async (req, res) => {
  const partnership = await partnershipService.createPartnership(req.body);
  res.json(partnership);
});

router.put('/:id', async (req, res) => {
  const partnership = await partnershipService.updatePartnership(req.params.id, req.body);
  res.json(partnership);
});

router.delete('/:id', async (req, res) => {
  await partnershipService.deletePartnership(req.params.id);
  res.json({ message: 'Partnership deleted successfully' });
});

export default router;
